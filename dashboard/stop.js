// "Stop" at the right end of every page's top bar: shuts the dashboard server down to free its
// memory. Recordings and lesson builds run in their own processes and keep going.
(() => {
  const bar = document.querySelector('header.topbar');
  if (!bar) return;
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'btn danger stop-dashboard';
  button.textContent = 'Stop';
  button.title = 'Stop the dashboard to free up memory. A recording in progress keeps going.';
  bar.appendChild(button);

  button.onclick = async () => {
    if (!confirm('Stop the dashboard?\n\nThis frees the memory it uses. A recording or lesson build in progress keeps going. Open it again from the project launcher.')) return;
    button.disabled = true;
    button.textContent = 'Stopping…';
    try {
      const response = await fetch('/api/dashboard/stop', {method: 'POST', headers: {'X-Notes-Dashboard': '1'}});
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || 'The dashboard could not be stopped.');
    } catch (error) {
      button.disabled = false;
      button.textContent = 'Stop';
      alert(error.message);
      return;
    }
    const notice = document.createElement('div');
    notice.className = 'dashboard-stopped';
    notice.setAttribute('role', 'status');
    notice.innerHTML = '<div><strong>Dashboard stopped.</strong><p>Its memory is freed. Open it again from the project launcher. You can close this tab.</p></div>';
    document.body.appendChild(notice);
    button.textContent = 'Stopped';
  };
})();
