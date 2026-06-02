#!/usr/bin/env node

/**
 * Transforma um arquivo JSON com um array de objetos em uma única linha
 * em um arquivo onde cada objeto ocupa sua própria linha.
 *
 * Usa streaming com buffer mínimo — apenas o objeto atual fica em memória,
 * independente do tamanho total do arquivo.
 *
 * Para arquivos .gz, descompacte antes: zcat input.json.gz | node format-json-array.js - output.json
 *
 * Uso:
 *   node scripts/format-json-array.js <entrada.json> <saida.json>
 *   node scripts/format-json-array.js - <saida.json>   # lê de stdin
 */

'use strict';

const fs = require('fs');
const zlib = require('zlib');
const path = require('path');

const [, , inputArg, outputFile] = process.argv;

if (!inputArg || !outputFile) {
  process.stderr.write(
    'Uso: node scripts/format-json-array.js <entrada.json[.gz]> <saida.json>\n' +
    '     node scripts/format-json-array.js - <saida.json>   (lê de stdin)\n'
  );
  process.exit(1);
}

// Seleciona fonte de leitura: stdin, .gz ou arquivo comum.
let readStream;
if (inputArg === '-') {
  readStream = process.stdin;
} else {
  const raw = fs.createReadStream(inputArg, { highWaterMark: 256 * 1024 });
  readStream = inputArg.endsWith('.gz') ? raw.pipe(zlib.createGunzip()) : raw;
}

readStream.setEncoding('utf8');

const writeStream = fs.createWriteStream(outputFile);

// Estado do parser de objetos JSON.
let pending = '';   // fragmento acumulado do objeto atual (ainda incompleto)
let depth = 0;      // profundidade de chaves abertas
let inString = false;
let escape = false;
let firstObject = true;
let count = 0;
const startTime = Date.now();

writeStream.write('[\n');

readStream.on('data', (chunk) => {
  // Concatena apenas o que sobrou do chunk anterior com o atual.
  // Em operação normal `pending` é vazio ou contém no máximo um objeto parcial.
  const text = pending + chunk;
  pending = '';
  let objStart = -1;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];

    // Consome o caractere escapado dentro de uma string.
    if (escape) {
      escape = false;
      continue;
    }

    // Dentro de string: só verifica escape e fechamento de string.
    if (inString) {
      if (c === '\\') escape = true;
      else if (c === '"') inString = false;
      continue;
    }

    // Fora de string: interpreta estrutura JSON.
    if (c === '"') {
      inString = true;
    } else if (c === '{') {
      if (depth === 0) objStart = i;
      depth++;
    } else if (c === '}') {
      depth--;
      if (depth === 0 && objStart !== -1) {
        // Objeto completo encontrado — emite.
        if (!firstObject) writeStream.write(',\n');
        writeStream.write(text.slice(objStart, i + 1));
        firstObject = false;
        count++;
        if (count % 100_000 === 0) {
          const s = ((Date.now() - startTime) / 1000).toFixed(1);
          process.stderr.write(`  ${count.toLocaleString()} objetos (${s}s)\n`);
        }
        objStart = -1;
      }
    }
  }

  // Guarda fragmento de objeto que cruzou o limite do chunk.
  if (objStart !== -1) {
    pending = text.slice(objStart);
  }
});

readStream.on('end', () => {
  writeStream.write('\n]\n');
  writeStream.end(() => {
    const s = ((Date.now() - startTime) / 1000).toFixed(1);
    process.stderr.write(`Concluído: ${count.toLocaleString()} objetos em ${s}s → ${outputFile}\n`);
  });
});

readStream.on('error', (err) => {
  process.stderr.write(`Erro de leitura: ${err.message}\n`);
  process.exit(1);
});

writeStream.on('error', (err) => {
  process.stderr.write(`Erro de escrita: ${err.message}\n`);
  process.exit(1);
});
