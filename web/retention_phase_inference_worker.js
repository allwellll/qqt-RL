'use strict';
// R5-only worker entry. Shared production worker and existing models are
// untouched; install the reference add-on before the first initialization.
importScripts('model_inference_worker.js', 'retention_phase_conditioning.js');
