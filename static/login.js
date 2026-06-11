function $(q){ return document.querySelector(q); }
function tabs(){ return Array.from(document.querySelectorAll('.tab')); }
function panes(){ return { login: $('#login'), register: $('#register') }; }

function switchTab(name){
  tabs().forEach(t => t.classList.toggle('active', t.dataset.tab===name));
  const p = panes();
  p.login.classList.toggle('active', name==='login');
  p.register.classList.toggle('active', name==='register');
  (name==='login' ? $('#login-email') : $('#reg-email')).focus();
}

async function api(path, payload){
  const r = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  const j = await r.json().catch(()=>({}));
  return { ok: r.ok, data: j };
}

async function doLogin(e){
  e.preventDefault();
  const email = $('#login-email').value.trim();
  const password = $('#login-pass').value;
  $('#login-msg').className='msg'; $('#login-msg').textContent='Signing in…';
  const { ok, data } = await api('/auth/login', { email, password });
  if(ok){
    localStorage.setItem('token', data.access_token);
    window.location.href = '/ui/chat';
  }else{
    $('#login-msg').className='msg error';
    $('#login-msg').textContent = data.detail || 'Login failed';
  }
  return false;
}

async function doRegister(e){
  e.preventDefault();
  const email = $('#reg-email').value.trim();
  const password = $('#reg-pass').value;
  $('#reg-msg').className='msg'; $('#reg-msg').textContent='Creating your account…';
  const { ok, data } = await api('/auth/register', { email, password });
  if(ok){
    $('#reg-msg').className='msg ok';
    $('#reg-msg').textContent = 'Registered! Please sign in.';
    switchTab('login');
    $('#login-email').value = email;
    $('#login-pass').focus();
  }else{
    $('#reg-msg').className='msg error';
    $('#reg-msg').textContent = data.detail || 'Could not register';
  }
  return false;
}

// boot
document.addEventListener('DOMContentLoaded', () => {
  const url = new URL(window.location.href);
  const start = url.searchParams.get('tab') || 'login';
  switchTab(start);
});