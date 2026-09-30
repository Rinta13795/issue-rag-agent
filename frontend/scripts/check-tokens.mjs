// 检查 src 中是否出现硬编码颜色：颜色值只允许写在 src/tokens.css（见仓库根目录 DESIGN.md）。
// 用法：npm run check:tokens
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../src/', import.meta.url))
const allowed = new Set(['tokens.css'])
const cssColor = /#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?|oklch|oklab|lab|lch|hwb)\(/
const tsColor = /['"`]#[0-9a-fA-F]{3,8}['"`]|\b(?:rgba?|hsla?|oklch)\(/

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name)
    return statSync(path).isDirectory() ? walk(path) : [path]
  })
}

const problems = []
for (const file of walk(root)) {
  const rel = relative(root, file)
  if (allowed.has(rel)) continue
  const pattern = file.endsWith('.css') ? cssColor : /\.(tsx?|jsx?)$/.test(file) ? tsColor : null
  if (!pattern) continue
  readFileSync(file, 'utf8').split('\n').forEach((line, index) => {
    if (pattern.test(line)) problems.push(`src/${rel}:${index + 1}  ${line.trim()}`)
  })
}

if (problems.length) {
  console.error('发现硬编码颜色，请改为引用 src/tokens.css 中的语义 token：\n' + problems.join('\n'))
  process.exit(1)
}
console.log('check:tokens 通过：颜色只出现在 src/tokens.css')
