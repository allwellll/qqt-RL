'use strict';

const REPEAT_TICKS = 4;

function steppedTick(current, direction, amount, maximum) {
  return Math.max(0, Math.min(maximum, current + direction * amount));
}

function isEditableTarget(target) {
  if (!target) return false;
  if (target.isContentEditable) return true;
  const tag = String(target.tagName || '').toLowerCase();
  if (['input', 'textarea', 'select'].includes(tag)) return true;
  if (target.getAttribute?.('contenteditable') === 'true' || target.getAttribute?.('role') === 'textbox') return true;
  return Boolean(target.closest?.('input, textarea, select, [contenteditable="true"], [role="textbox"]'));
}

function bindReviewKeyboard({ onStep }) {
  return event => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') {
      return { handled: false, prevented: false };
    }
    if (isEditableTarget(event.target)) return { handled: false, prevented: false };
    const direction = event.key === 'ArrowLeft' ? -1 : 1;
    const amount = event.repeat ? REPEAT_TICKS : 1;
    event.preventDefault();
    onStep(direction, amount);
    return { handled: true, prevented: true };
  };
}

const api = { bindReviewKeyboard, REPEAT_TICKS, steppedTick };
if (typeof module !== 'undefined') module.exports = api;
if (typeof window !== 'undefined') window.ReviewKeyboard = api;
