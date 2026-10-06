// Execute the exact generated script with a minimal DOM in Node; no browser claim.
const fs=require('node:fs'), path=require('node:path'), vm=require('node:vm'), assert=require('node:assert/strict');
const bundle=path.resolve(process.argv[2]), output=path.resolve(process.argv[3]);
const overall=process.argv.includes('--overall');
assert(!fs.existsSync(output),'Refuse to overwrite validation');
const results=[];
for(const group of ['gender','religion','regional']) {
  const page=fs.readFileSync(path.join(bundle,group,'explorer.html'),'utf8');
  const payload=page.split('<script id="payload" type="application/json">')[1].split('</script>')[0];
  const script=page.split('</script><script>')[1].split('</script>')[0];
  const elements=[];
  class Element {
    constructor(tag){this.tag=tag;this.children=[];this.value='';this.style={};this._text='';elements.push(this);}
    appendChild(child){this.children.push(child);return child;}
    add(option){this.children.push(option);}
    replaceChildren(){this.children=[];}
    set textContent(text){this._text=String(text);}
    get textContent(){return this._text+this.children.map(c=>c.textContent||'').join('');}
    click(){if(!this.disabled&&this.onclick)this.onclick();}
  }
  const ids=Object.fromEntries(['payload','filters','status','prev','next'].map(k=>[k,new Element('div')]));
  ids.payload.textContent=payload;
  const head=new Element('thead'),body=new Element('tbody');
  const document={getElementById:id=>ids[id],createElement:tag=>new Element(tag),querySelector:q=>q==='thead'?head:q==='tbody'?body:body.children[0],querySelectorAll:q=>q==='select'?elements.filter(e=>e.tag==='select'):body.children};
  const context=vm.createContext({document,Option:class {constructor(text,value){this.textContent=text;this.value=value;}}});
  vm.runInContext(script,context,{timeout:30000});
  const evaluate=code=>vm.runInContext(code,context,{timeout:30000});
  const total=evaluate('data.length');
  if(overall) assert(evaluate("!filters.subset && data.every(r=>r[cols.indexOf('subset')]==='ethics_all')"));
  assert.equal(evaluate('filtered.length'),group==='gender'?24:144);
  assert.equal(body.children.length,group==='gender'?24:80);
  evaluate("Object.values(filters).forEach(s=>s.value='');page=0;render();");
  const first=body.children[0].textContent;
  ids.next.click();assert.notEqual(first,body.children[0].textContent);
  evaluate("filters.decision.value='not_significant';page=0;render();");
  assert(evaluate("filtered.length>0 && filtered.every(r=>r[cols.indexOf('decision')]==='not_significant')"));
  evaluate("filters.decision.value='not_testable';page=0;render();");
  const unavailable=evaluate('filtered.length');if(!overall) assert(unavailable>0);
  assert.equal(unavailable,evaluate("data.filter(r=>r[cols.indexOf('decision')]==='not_testable').length"));
  assert(evaluate("filtered.every(r=>r[cols.indexOf('n_pairs')]===0 || r[cols.indexOf('p_holm')]===null)"));
  evaluate("filters.decision.value='';if(filters.subset) filters.subset.value='virtue';filters.comparison.value='vs_explicit';filters.scale.value='numeric';page=0;render();");
  assert(evaluate(`filtered.length>0 && filtered.every(r=>r[cols.indexOf('subset')]==='${overall?'ethics_all':'virtue'}' && r[cols.indexOf('comparison')]==='vs_explicit' && r[cols.indexOf('scale')]==='numeric')`));
  for(const filename of ['all_comparisons.csv',overall?'figure_index.csv':'heterogeneity.csv','source_manifest.json']) assert(fs.existsSync(path.join(bundle,group,filename)));
  results.push({group,total,unavailable,defaultFilter:true,pagination:true,nonSignificantFilter:true,missingFilter:true,combinedFilters:true,linksExist:true,overall});
  console.log(JSON.stringify(results.at(-1)));
}
fs.writeFileSync(output,JSON.stringify({valid:true,engine:'Node vm with minimal DOM; not a browser layout test',results},null,2));
