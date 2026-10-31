'use strict';
function validate(pair){
 if(!Array.isArray(pair)||pair.length!==2||pair.some(p=>!Array.isArray(p)||p.length!==2||p.some(x=>!Number.isInteger(x)))||pair.some(p=>p[0]<0||p[0]>=13||p[1]<0||p[1]>=15)||pair[0][1]>6||pair[1][1]<8)throw Error('spawn side must match physical team; center column excluded');
 return pair;
}
function apply(sim,pair){validate(pair);for(let p=0;p<2;p++){sim.pos[2*p]=pair[p][0]+.5;sim.pos[2*p+1]=pair[p][1]+.5;}sim.bunSpawnPos=pair.map(p=>p.map(x=>x+.5));sim._gen++;}
module.exports={VERSION:'own-side-native-v1',validate,apply};
