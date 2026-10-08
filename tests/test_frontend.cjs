const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {foldIn,preprocess}=require('../dist/app.js');
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'../dist/assets/model-data.json'),'utf8'));
assert.deepEqual(preprocess('The SPACE station, and 123 rockets!'),['space','station','rockets']);
for(const doc of data.documents)assert.deepEqual(preprocess(doc.text),doc.tokens,'Preprocessing mismatch: '+doc.id);
assert.equal(foldIn('Unicorns dance fabulously.',data).status,'no_vocabulary');
assert.throws(()=>foldIn('',data));assert.throws(()=>foldIn('a'.repeat(5001),data));
for(const [text,target] of [['The spacecraft entered orbit around the planet during the space mission.','Space exploration'],['The doctor examined the patient at the hospital and prescribed medicine for treatment.','Healthcare'],['The cricket team scored runs in the final match.','Sport and cricket'],['The software developer wrote code for a computer program.','Computing and software']]){
 const result=foldIn(text,data),total=result.theta.reduce((a,b)=>a+b,0);assert.ok(Math.abs(total-1)<1e-10);assert.ok(result.theta.every(v=>v>0&&v<1));assert.equal(data.topics[result.theta.indexOf(Math.max(...result.theta))].label,target);assert.deepEqual(result,foldIn(text,data));
}
console.log('PASS: preprocessing, unknown vocabulary, validation, normalized fold-in, expected topics and deterministic results.');
