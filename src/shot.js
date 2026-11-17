const { chromium } = require('playwright');
const items=[['01-landing',1440,900],['02-analyzing',1440,800],['03-dashboard',1440,900],['04-explore-drawer',1440,940],['05-ask-the-comments',1440,900],['06-mobile',390,844],['07-design-system',1360,900]];
(async()=>{const b=await chromium.launch();
for(const [n,w,h] of items){const p=await b.newPage({viewport:{width:w,height:h},deviceScaleFactor:2});
await p.goto('file:///tmp/outputs/ui-kit/src/'+n+'.html');await p.waitForTimeout(300);
await p.screenshot({path:'/tmp/outputs/ui-kit/'+n+'.png',fullPage:true});}
await b.close();})();
