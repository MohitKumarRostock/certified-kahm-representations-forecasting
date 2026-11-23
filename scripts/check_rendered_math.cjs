#!/usr/bin/env node
"use strict";
/* Validate every Markdown/notebook formula with KaTeX and MathJax. */
const fs = require("node:fs");
const path = require("node:path");
const katex = require("katex");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");
const ROOT = path.resolve(__dirname, "..");
const SKIP = new Set([".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build"]);
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const mj = mathjax.document("", { InputJax: new TeX({ packages: AllPackages }), OutputJax: new SVG({ fontCache: "none" }) });
function scanFiles(folder, results = []) {
  for (const item of fs.readdirSync(folder, { withFileTypes: true })) {
    if (item.isDirectory()) { if (!SKIP.has(item.name)) scanFiles(path.join(folder,item.name),results); }
    else if (item.isFile() && /\.(md|ipynb)$/i.test(item.name)) results.push(path.join(folder,item.name));
  }
  return results;
}
function extractMath(markdown,file) {
  const results=[],lines=markdown.split(/\r?\n/); let fence=null,block=null;
  for(let i=0;i<lines.length;i++){
    const line=lines[i],fenceMark=line.match(/^\s*(\x60{3,}|~{3,})/);
    if(fenceMark){
      const marker=fenceMark[1],language=line.slice(line.indexOf(marker)+marker.length).trim();
      if(fence===null){
        fence={char:marker[0],size:marker.length,isMath:language==="math",line:i+1,parts:[]};
      }else if(fence.char===marker[0]&&marker.length>=fence.size){
        if(fence.isMath)results.push({file,line:fence.line,latex:fence.parts.join("\n"),display:true,mode:"fenced"});
        fence=null;
      }
      continue;
    }
    if(fence!==null){if(fence.isMath)fence.parts.push(line);continue;}
    if(/^\s*\$\$\s*$/.test(line)){
      if(block===null)block={line:i+1,parts:[]};
      else{results.push({file,line:block.line,latex:block.parts.join("\n"),display:true,mode:"dollars"});block=null;}
      continue;
    }
    if(block!==null){block.parts.push(line);continue;}
    // GitHub's protected inline math form uses a dollar/backtick pair.
    const protectedMath=/\$\x60([^\x60]+)\x60\$/g; let match;
    while((match=protectedMath.exec(line))!==null)
      results.push({file,line:i+1,latex:match[1],display:false,mode:"protected"});
    const withoutProtected=line.replace(protectedMath," ");
    const plain=withoutProtected.replace(/\x60+[^\x60]*\x60+/g," ");
    const inline=/(?<!\\)\$((?:\\.|[^$])+?)(?<!\\)\$/g; let found;
    while((found=inline.exec(plain))!==null)results.push({file,line:i+1,latex:found[1],display:false,mode:"inline"});
  }
  if(block!==null)results.push({file,line:block.line,latex:block.parts.join("\n"),display:true,mode:"dollars",unterminated:true});
  if(fence!==null&&fence.isMath)results.push({file,line:fence.line,latex:fence.parts.join("\n"),display:true,mode:"fenced",unterminated:true});
  return results;
}
const mistakes=[];let mathCount=0,fileCount=0;
for(const abs of scanFiles(ROOT)){
  const relative=path.relative(ROOT,abs).replaceAll(path.sep,"/");const markdowns=[];
  if(abs.endsWith(".ipynb")){
    let notebook;
    try{notebook=JSON.parse(fs.readFileSync(abs,"utf8"));}
    catch(err){mistakes.push(relative+": invalid notebook JSON: "+err.message);continue;}
    for(const [index,cell] of (notebook.cells||[]).entries())if(cell.cell_type==="markdown"){
      const source=Array.isArray(cell.source)?cell.source.join(""):(cell.source||"");
      markdowns.push([source,relative+"#cell-"+(index+1)]);
    }
  }else markdowns.push([fs.readFileSync(abs,"utf8"),relative]);
  fileCount++;
  for(const [markdown,label] of markdowns)for(const expr of extractMath(markdown,label)){
    mathCount++;const prefix=expr.file+":"+expr.line;
    if(expr.unterminated)mistakes.push(prefix+": unclosed display equation");
    // Direct TeX parsing does not reproduce Markdown's punctuation-unescaping step.
    // Prefer fenced math and protected inline math for sensitive source.
    const fragile=/\\[{}!,;:|]|\*|\\(?:left|right)\b|\\\\/;
    if((expr.mode==="inline"||expr.mode==="dollars")&&fragile.test(expr.latex))
      mistakes.push(prefix+": Markdown-sensitive TeX must use a math fence or protected inline math");
    if(/\\(?:operatorname\*?|tag|hbox|rm)(?![a-zA-Z])/.test(expr.latex))mistakes.push(prefix+": unsupported GitHub macro");
    try{katex.renderToString(expr.latex,{displayMode:expr.display,throwOnError:true,strict:"ignore",trust:false,output:"html"});}
    catch(err){mistakes.push(prefix+": KaTeX: "+err.message+"\n  "+expr.latex.slice(0,350).replaceAll("\n"," "));}
    try{
      const html=adaptor.outerHTML(mj.convert(expr.latex,{display:expr.display}));
      const error=html.match(/data-mjx-error="([^"]+)"/);
      if(error)mistakes.push(prefix+": MathJax: "+error[1]+"\n  "+expr.latex.slice(0,350).replaceAll("\n"," "));
    }catch(err){mistakes.push(prefix+": MathJax exception: "+err.message);}
  }
}
if(mistakes.length){
  console.error(mistakes.join("\n"));
  console.error("Math rendering check FAILED: "+mistakes.length+" issue(s) across "+mathCount+" expressions.");
  process.exit(1);
}
console.log("Math rendering check passed: "+mathCount+" expressions in "+fileCount+" Markdown/notebook files.");
