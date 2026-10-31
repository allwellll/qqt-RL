import numpy as np
import jax.numpy as j
from scripts.bomb_threat_full_signal_audit import attribution

def test_proportional_shadow_conserves_capped_payment_and_cancelled_credit():
    pending=j.array([[.01,.02,0.]])
    offers=j.array([[.02,.04,.01]])
    remaining,pay,cancel=attribution(pending,offers,j.array([.035]),j.array([0.]),j.array([.065]))
    np.testing.assert_allclose(remaining,0,atol=1e-7)
    np.testing.assert_allclose(pay.sum(-1),[.065],atol=1e-7)
    remaining,pay,cancel=attribution(pending,j.zeros_like(offers),j.array([0.]),j.array([.03]),j.array([0.]))
    np.testing.assert_allclose(cancel,pending,atol=1e-7);np.testing.assert_allclose(remaining,0,atol=1e-7)
