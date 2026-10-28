'use strict';
// Invoke the unchanged diagnostics entry after passive preloading cached it.
const [model, out, regions, smoke] = process.argv.slice(2);
require('./je08_web_diagnostics.js').run(model, out, regions, smoke === 'smoke')
  .catch(error => { console.error(error); process.exitCode = 1; });
