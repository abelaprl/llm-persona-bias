// Exercise the local offline explorer in an isolated headless Edge profile.
const fs = require('node:fs');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {pathToFileURL} = require('node:url');
const assert = require('node:assert/strict');

async function main() {
  const bundle = path.resolve(process.argv[2]);
  const output = path.resolve(process.argv[3]);
  const overall = process.argv.includes('--overall');
  fs.mkdirSync(output);
  const profile = path.join(output, 'edge_profile');
  const edge = spawn('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', [
    '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    '--disable-background-networking', '--disable-extensions', '--remote-debugging-port=0',
    `--user-data-dir=${profile}`, 'about:blank'
  ], {windowsHide: true, stdio: 'ignore'});
  const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
  let socket;
  try {
    const active = path.join(profile, 'DevToolsActivePort');
    for (let n = 0; !fs.existsSync(active) && n < 100; n++) await delay(200);
    assert(fs.existsSync(active), 'Headless browser did not start');
    const port = fs.readFileSync(active, 'utf8').split('\n')[0];
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    socket = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {socket.onopen=resolve;socket.onerror=reject;});
    let sequence=0;
    const pending = new Map();
    socket.onmessage = event => {
      const message=JSON.parse(event.data);
      if(pending.has(message.id)) {
        const {resolve,reject}=pending.get(message.id);
        pending.delete(message.id);
        message.error?reject(message.error):resolve(message.result);
      }
    };
    const call = (method, params={}) => new Promise((resolve,reject) => {
      const id=++sequence;
      pending.set(id,{resolve,reject});
      socket.send(JSON.stringify({id,method,params}));
    });
    const evaluate = async expression => {
      const response=await call('Runtime.evaluate',{expression,returnByValue:true});
      assert(!response.exceptionDetails,JSON.stringify(response.exceptionDetails));
      return response.result.value;
    };
    await call('Page.enable');
    await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    const results=[];
    for(const group of ['gender','religion','regional']) {
      await call('Page.navigate',{url:pathToFileURL(path.join(bundle,group,'explorer.html')).href});
      let status;
      for(let n=0;n<150;n++) {
        status=await evaluate("document.getElementById('status')?.textContent || ''");
        if(status.includes('hasil cocok')) break;
        await delay(200);
      }
      assert(status.includes('hasil cocok'),`${group}: page did not render`);
      const initial=await evaluate("({rows:data.length,matched:filtered.length,rendered:document.querySelectorAll('tbody tr').length,filters:document.querySelectorAll('select').length})");
      assert.equal(initial.filters,overall?9:10);
      if(overall) assert(await evaluate("!filters.subset && data.every(r=>r[cols.indexOf('subset')]==='ethics_all')"));
      assert.equal(initial.matched,group==='gender'?24:144);
      assert.equal(initial.rendered,Math.min(initial.matched,80));
      await evaluate("Object.values(filters).forEach(s=>s.value='');page=0;render();true");
      const first=await evaluate("document.querySelector('tbody tr').textContent");
      await evaluate("document.getElementById('next').click();true");
      const second=await evaluate("document.querySelector('tbody tr').textContent");
      assert.notEqual(first,second);
      await evaluate("filters.decision.value='not_significant';page=0;render();true");
      assert(await evaluate("filtered.length>0 && filtered.every(r=>r[cols.indexOf('decision')]==='not_significant')"));
      const missingCount=await evaluate("data.filter(r=>r[cols.indexOf('decision')]==='not_testable').length");
      if(!overall) assert(missingCount>0);
      if(missingCount) {
        await evaluate("filters.decision.value='not_testable';page=0;render();true");
        assert.equal(missingCount,await evaluate("filtered.length"));
        assert(await evaluate("filtered.every(r=>r[cols.indexOf('n_pairs')]===0 || r[cols.indexOf('p_holm')]===null)"));
      } else {
        assert(await evaluate("![...filters.decision.options].some(o=>o.value==='not_testable')"));
      }
      await evaluate("filters.decision.value='';if(filters.subset) filters.subset.value='commonsense';filters.comparison.value='direct';filters.outcome_id.value='word_count';page=0;render();true");
      const screenshot=await call('Page.captureScreenshot',{format:'png'});
      fs.writeFileSync(path.join(output,`${group}_explorer.png`),Buffer.from(screenshot.data,'base64'));
      results.push({group,...initial,missingCount,pagination:true,nonSignificantFilter:true,naFilter:missingCount?'tested':'no_missing_rows_option_absent',overall});
      console.log(group,JSON.stringify(results.at(-1)));
    }
    fs.writeFileSync(path.join(output,'validation.json'),JSON.stringify({valid:true,results},null,2));
    await call('Browser.close');
  } finally {
    if(socket) socket.close();
    edge.kill();
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
