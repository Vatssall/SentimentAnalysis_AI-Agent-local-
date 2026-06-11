let sessionId = null;
let sessions = [];

function token(){ return localStorage.getItem('token'); }
function authHeaders(){ return {'Content-Type':'application/json', 'Authorization':'Bearer ' + token()}; }
function guard(){ if(!token()) location.href = '/ui/login'; }
guard();

function qs(x){ return document.querySelector(x); }
function el(tag, cls){ const e=document.createElement(tag); if(cls) e.className=cls; return e; }

async function api(path, opts={}){
  const r = await fetch(path, { ...opts, headers: { ...(opts.headers||{}), ...authHeaders() } });
  const j = await r.json().catch(()=>({}));
  if(!r.ok) throw new Error(j.detail || 'Request failed');
  return j;
}

// -------- sessions ----------
async function loadSessions(){
  const j = await api('/sessions');
  sessions = j.items || [];
  renderSessions();
  if(!sessionId && sessions.length){ selectSession(sessions[0].id); }
}

function renderSessions(){
  const box = qs('#sessions'); box.innerHTML='';
  sessions.forEach(s=>{
    const row = el('div','session'+(s.id===sessionId?' active':''));
    const left = el('div'); const right = el('div','actions');
    const title = el('div','title'); title.textContent = s.title || 'Chat';
    const preview = el('div','preview'); preview.textContent = s.preview || '';
    left.appendChild(title); left.appendChild(preview);

    const rename = el('button'); rename.textContent='✎'; rename.title='Rename';
    rename.onclick=()=>promptRename(s);
    const del = el('button'); del.textContent='🗑'; del.title='Delete';
    del.onclick=()=>deleteSession(s.id);

    right.appendChild(rename); right.appendChild(del);
    row.appendChild(left); row.appendChild(right);
    row.onclick = (e)=>{ if(e.target.tagName==='BUTTON') return; selectSession(s.id); };
    box.appendChild(row);
  });
}

async function newChat(){
  const j = await api('/sessions', {method:'POST'});
  sessionId = j.id; qs('#chat-title').value = j.title || 'New chat';
  await loadMessages(); await loadSessions();
}

async function deleteSession(id){
  if(!confirm('Delete this chat?')) return;
  await api(`/sessions/${id}`, {method:'DELETE'});
  if(id===sessionId) sessionId=null;
  await loadSessions(); qs('#chat').innerHTML='';
}

function promptRename(s){
  const t = prompt('Rename chat', s.title || '');
  if(t && t.trim()) renameSession(s.id, t.trim());
}

async function renameSession(id, title){
  await api(`/sessions/${id}`, {method:'PATCH', body: JSON.stringify({title})});
  if(id===sessionId) qs('#chat-title').value = title;
  await loadSessions();
}

async function renameActive(){
  if(!sessionId) return;
  await renameSession(sessionId, qs('#chat-title').value.trim() || 'Chat');
}

async function selectSession(id){
  sessionId = id;
  const s = sessions.find(x=>x.id===id);
  qs('#chat-title').value = (s?.title)||'Chat';
  await loadMessages();
  renderSessions();
}

// -------- messages ----------
function renderMessage(m){
  const tpl = qs('#msg-template').content.cloneNode(true);
  const node = tpl.querySelector('.msg');

  node.classList.add(m.role);
  node.querySelector('.content').textContent = m.content;
  node.querySelector('.time').textContent = new Date(m.created_at || Date.now()).toLocaleTimeString();

  const chips = node.querySelector('.chips');

  if (m.signals) {
    const s = m.signals;
    const mk = (name, obj, cls) => {
      if (!obj || !obj.label) return;
      const c = el('span', 'chip ' + cls);
      const p = (typeof obj.proba === 'number') ? ` (p=${obj.proba.toFixed(2)})` : '';
      c.textContent = `${name}: ${obj.label}${p}`;
      chips.appendChild(c);
    };
    mk('sentiment', s.sentiment, 'sentiment');
    mk('emotion',   s.emotion,   'emotion');
    mk('mood',      s.mood,      'mood');
    if (chips.children.length === 0) chips.remove();
  } else {
    chips.remove();
  }

  qs('#chat').appendChild(node);
  qs('#chat').scrollTop = qs('#chat').scrollHeight;
  return node;
}

function renderInlineTasks(parentMsgNode, tasks){
  if(!tasks || !tasks.length) return;
  const wrap = el('div');
  wrap.style.marginTop = '10px';
  tasks.forEach(t=>{
    const card = el('div','task');
    card.innerHTML = `
      <b>${t.title}</b> <span class="mins">${t.minutes}m</span>
      <p>${t.description}</p>
    `;
    wrap.appendChild(card);
  });
  parentMsgNode.querySelector('.bubble').appendChild(wrap);
}

async function loadMessages(){
  if(!sessionId) return;
  qs('#chat').innerHTML='';
  const j = await api(`/sessions/${sessionId}/messages`);
  (j.items||[]).forEach(renderMessage);
}

// -------- chat send ----------
async function send(){
  const box = qs('#input');
  const t = box.value.trim(); if(!t) return;
  box.value='';
  renderMessage({role:'user', content:t, created_at: new Date().toISOString()});
  qs('#status').textContent='Thinking…';
  try{
    const j = await api('/chat', {
      method:'POST',
      body: JSON.stringify({ text: t, session_id: sessionId })
    });
    sessionId = j.session_id;
    const node = renderMessage({role:'assistant', content:j.reply, created_at:new Date().toISOString(), signals:j.signals});
    renderInlineTasks(node, j.tasks);  // show therapist exercises (if any)
    await loadSessions(); // refresh titles/previews
  }catch(e){
    renderMessage({role:'assistant', content:'Error: '+e.message, created_at:new Date().toISOString()});
  }finally{
    qs('#status').textContent='';
  }
}

function logout(){ localStorage.removeItem('token'); location.href='/ui/login'; }

// boot
loadSessions();