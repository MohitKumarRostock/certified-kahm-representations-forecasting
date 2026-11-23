#!/usr/bin/env node
"use strict";
/**
 * Validate the math that GitHub's Markdown renderer actually emits.
 * Raw LaTeX checks alone miss GitHub/HTML entity conversion and markup splitting.
 */
const fs=require("node:fs"),path=require("node:path");
const {mathjax}=require("mathjax-full/js/mathjax.js");
const {TeX}=require("mathjax-full/js/input/tex.js");
const {SVG}=require("mathjax-full/js/output/svg.js");
const {liteAdaptor}=require("mathjax-full/js/adaptors/liteAdaptor.js");
const {RegisterHTMLHandler}=require("mathjax-full/js/handlers/html.js");
const {AllPackages}=require("mathjax-full/js/input/tex/AllPackages.js");
const adaptor=liteAdaptor();RegisterHTMLHandler(adaptor);
const mj=mathjax.document("",{InputJax:new TeX({packages:AllPackages}),OutputJax:new SVG({fontCache:"none"})});
const ROOT=path.resolve(__dirname,".."),REPO=process.env.GITHUB_REPOSITORY||"MohitKumarRostock/certified-kahm-representations-forecasting";
const ROOTS=[ROOT, path.join(ROOT,"docs"), path.join(ROOT,"reports")];
const files=[];
for(const folder of ROOTS)if(fs.existsSync(folder)){
  for(const entry of fs.readdirSync(folder,{withFileTypes:true})){
    if(entry.isFile()&&entry.name.endsWith(".md"))files.push(path.join(folder,entry.name));
    if(folder.endsWith("/reports")&&entry.isDirectory()&&entry.name==="lower_certificates_v1"){
      for(const name of fs.readdirSync(path.join(folder,entry.name)))if(name.endsWith(".md"))files.push(path.join(folder,entry.name,name));
    }
  }
}
// HTML text inside <math-renderer> is entity-decoded exactly once by the browser.
// An entity like &amp;gt; becomes the literal string &gt; in textContent.
function decodeOnce(s){
  return s.replace(/&(#(?:x[0-9A-Fa-f]+|\d+)|amp|lt|gt|quot|apos|nbsp);/g,(whole,name)=>{
    if(name==="amp")return "&";if(name==="lt")return "<";if(name==="gt")return ">";
    if(name==="quot")return '"';if(name==="apos")return "'";if(name==="nbsp")return "\u00a0";
    if(name.startsWith("#x"))return String.fromCodePoint(parseInt(name.slice(2),16));
    if(name.startsWith("#"))return String.fromCodePoint(parseInt(name.slice(1),10));
    return whole;
  });
}
async function gitHubHtml(text){
  const res=await fetch("https://api.github.com/markdown",{method:"POST",headers:{
    "Accept":"text/html","Content-Type":"application/json",
    "X-GitHub-Api-Version":"2022-11-28","User-Agent":"JMLR-equation-validation",
    ...(process.env.GITHUB_TOKEN?{"Authorization":"Bearer "+process.env.GITHUB_TOKEN}:{})
  },body:JSON.stringify({text,mode:"gfm",context:REPO})});
  if(!res.ok)throw Error("GitHub Markdown API HTTP "+res.status+": "+(await res.text()).slice(0,250));
  return await res.text();
}
let count=0,failures=[],filesChecked=0;
for(const file of files){
  const rel=path.relative(ROOT,file).replaceAll(path.sep,"/");
  const html=await gitHubHtml(fs.readFileSync(file,"utf8"));
  let index=0,fileErr=0;
  const items=[...html.matchAll(/<math-renderer\b([^>]*)>([\s\S]*?)<\/math-renderer>/g)];
  for(const item of items){
    const original=decodeOnce(item[2]);
    const display=item[1].includes("js-display-math");
    let latex=original.trim();
    if(latex.startsWith("$$")&&latex.endsWith("$$"))latex=latex.slice(2,-2);
    else if(latex.startsWith("$")&&latex.endsWith("$"))latex=latex.slice(1,-1);
    count++;index++;
    const output=adaptor.outerHTML(mj.convert(latex,{display}));
    const error=(output.match(/data-mjx-error="([^"]+)"/)||[])[1];
    if(error){
      failures.push({path:rel,index,error,latex:latex.replace(/\s+/g," ").slice(0,240)});
      fileErr++;
    }
  }
  filesChecked++;
  console.log("FILE",rel,"MATH",items.length,"ERRORS",fileErr);
}
console.log("RENDERED HTML MATH AUDIT",JSON.stringify({filesChecked,expressions:count,errors:failures.length}));
for(const entry of failures.slice(0,90))console.log("ERROR",JSON.stringify(entry));
if(failures.length){process.exitCode=1;}
