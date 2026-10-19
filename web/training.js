'use strict';
(async()=>{
  const status=document.querySelector('#status'),body=document.querySelector('#results'),select=document.querySelector('#round');
  try {
    const response=await fetch('training-summary.json',{cache:'no-cache'});
    if(!response.ok)throw new Error('HTTP '+response.status);
    const data=await response.json();
    if(data.schema!=='je10.public-training-summary/v1'||data.status!=='PAUSED'||data.successor_started||data.rounds.reduce((n,r)=>n+r.points.length,0)!==data.verified_points)throw new Error('结果身份不一致');
    status.textContent=`${data.date} · ${data.status} · g${data.paused_generation} 原子暂停 · ${data.verified_points} 个完整固定点`;
    for(const round of data.rounds){const option=document.createElement('option');option.value=String(round.round);option.textContent=`r${round.round} · ${round.status}`;select.append(option);}
    function render(){body.replaceChildren();for(const r of data.rounds){if(select.value!=='all'&&select.value!==String(r.round))continue;for(const p of r.points){const row=document.createElement('tr');for(const text of [`r${r.round} g${p.generation}`,p.hunter.join('/'),`${p.tactical}/64`,p.deliveries.join('/'),`${p.natural_pickup}/16`,String(p.tactical_self)]){const cell=document.createElement('td');cell.textContent=text;row.append(cell);}body.append(row);}}}
    select.addEventListener('change',render);render();
  } catch(error){status.textContent='冻结结果加载失败：'+error.message;}
})();
