"use strict";
const { chromium } = require("playwright-core");
const fs = require("node:fs");
const pages = [
  ["final-proof", "https://github.com/MohitKumarRostock/certified-kahm-representations-forecasting/blob/7710b3eb18ee40859026866169379896a009d556/docs/LOWER_CERTIFICATE_PROOF.md"],
  ["final-experiments", "https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/tree/54b820a35e7ece691734effd3bd93d01e938204e"],
  ["final-experiment-guide", "https://github.com/MohitKumarRostock/Kernel_Affine_Hull_Koopman_Machine/blob/54b820a35e7ece691734effd3bd93d01e938204e/docs/EXPERIMENT_GUIDE.md"],
];
(async()=>{
  const browser=await chromium.launch({channel:"chrome",headless:true,args:["--no-sandbox","--disable-dev-shm-usage"]});
  for(const [label,url] of pages){
    const page=await browser.newPage({viewport:{width:1440,height:900}});
    const errors=[];
    page.on("pageerror",err=>errors.push("pageerror "+String(err).slice(0,500)));
    page.on("console",msg=>{if(msg.type()==="error")errors.push("console "+msg.text().slice(0,500))});
    try{
      const resp=await page.goto(url,{waitUntil:"domcontentloaded",timeout:85000});
      console.log("PAGE",label,"http",resp.status(),"final",page.url());
      await page.waitForFunction(()=>document.querySelectorAll("math-renderer").length>0,{timeout:25000}).catch(()=>{});
      await page.waitForTimeout(6500);
      const report=await page.evaluate(()=>{
        const items=[...document.querySelectorAll("math-renderer")];
        return{
          title:document.title,
          count:items.length,
          errors:items.map((el,i)=>{
            const dom=el.shadowRoot||el;
            const txt=(dom.innerText||dom.textContent||"").trim();
            const selectors=["mjx-merror","merror",".katex-error","[data-mjx-error]",".error"];
            const any=selectors.flatMap(q=>[...dom.querySelectorAll(q)].map(e=>({selector:q,text:(e.textContent||"").slice(0,500)})));
            const bad=/Extra \\left|Missing \\right|Missing close brace|Missing open brace|Misplaced &|Undefined control sequence|Math Processing Error|unknown command|unrecognized delimiter/i.test(txt);
            if(!bad&&!any.length)return null;
            return{index:i,source:(el.getAttribute("data-tex")||el.textContent||"").slice(0,400),
              bad,errors:any.slice(0,5),text:txt.slice(0,450),html:dom.innerHTML.slice(0,700)};
          }).filter(Boolean).slice(0,80),
          githubMessage:[...document.querySelectorAll(".flash-error")].map(e=>e.textContent.slice(0,300))
        };
      });
      console.log("RESULT",label,JSON.stringify(report).slice(0,28000));
      fs.mkdirSync("/tmp/ghmath",{recursive:true});
      await page.screenshot({path:"/tmp/ghmath/"+label+".png",fullPage:false});
    }catch(err){console.log("FAIL",label,String(err))}
    console.log("BROWSER_ERRORS",label,JSON.stringify(errors.slice(0,15)));
    await page.close();
  }
  await browser.close();
})().catch(e=>{console.error(e);process.exitCode=1});
