/**
 * Entry point. Import the application as a dynamic module and run it. A failed module load or a
 * failed bundle fetch leaves the loading screen on a short message with a Reload button.
 *
 * @module main
 */

/** Replace the loading screen content with the failure message and a Reload button. */
function fail(err) {
  console.error(err);
  const l = document.getElementById('loading');
  if (!l) return;
  l.classList.add('is-error');
  l.textContent = '';
  const p = document.createElement('p');
  p.textContent = 'Could not load the page. Reload, or start the server from the site folder: ' +
    'cd site && python -m http.server 8000';
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'btn';
  b.textContent = 'Reload';
  b.addEventListener('click', () => window.location.reload());
  l.append(p, b);
}

import('./app.js').then((app) => app.main()).catch(fail);
