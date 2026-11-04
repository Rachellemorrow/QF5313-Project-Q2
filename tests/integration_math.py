"""Lead verification: independently formulated deterministic and optimization oracles."""
import numpy as np
import torch
from scipy.optimize import minimize_scalar

from mc_common import Model, gbm_paths_from_z_np, gbm_paths_from_z_torch, geometric_asian_price
from approach2_hedging import empirical_cvar, hedging_pnl_np, hedging_pnl_torch, PolicyMLP, neural_delta, evaluate_neural, _neural_losses


def test_exact_path_identity_and_geometric_covariance():
    model = Model(S0=97, K=102, r=.017, sigma=.24, T=1, steps=252)
    z = np.random.default_rng(813).normal(size=(16,252))
    reference = np.empty((16,253))
    reference[:,0] = model.S0
    for j in range(252):
        reference[:,j+1] = reference[:,j]*np.exp((model.r-model.sigma**2/2)/252+model.sigma/np.sqrt(252)*z[:,j])
    np.testing.assert_allclose(gbm_paths_from_z_np(model,z),reference,rtol=1e-13)
    np.testing.assert_allclose(gbm_paths_from_z_torch(model,z).numpy(),reference,rtol=1e-13)
    times = np.arange(1,253)/252
    _, a, b = geometric_asian_price(model.S0,model.K,model.r,model.sigma,1,252)
    assert abs(b-model.sigma**2*np.minimum.outer(times,times).mean()) < 1e-15
    assert abs(a-(np.log(model.S0)+(model.r-model.sigma**2/2)*times.mean())) < 1e-14


def test_literal_pnl_irregular_final_interval_and_cost_sign():
    model = Model(S0=100,K=103,r=.09,sigma=.2,T=7/252,steps=7)
    prices = np.array([[100,99,101,104,102,106,105,110], [100,101,102,103,104,105,106,107]], dtype=float)
    indices = np.array([0,5])
    holdings = np.array([[.2,.8],[-.5,.25]])
    expected = []
    for s,d in zip(prices,holdings):
        gains = np.exp(model.r*model.T)*d[0]*(s[5]-s[0])+np.exp(model.r*(model.T-5/252))*d[1]*(s[7]-s[5])
        costs = .005*(abs(d[0])*s[0]+abs(d[1]-d[0])*s[5])
        expected.append(gains-costs-max(s[-1]-103,0))
    actual,turnover,cost = hedging_pnl_np(prices,holdings,indices,.005,model)
    np.testing.assert_allclose(actual,expected,rtol=1e-13)
    np.testing.assert_allclose(hedging_pnl_torch(torch.tensor(prices),torch.tensor(holdings),indices,.005,model).numpy(),expected,rtol=1e-13)
    np.testing.assert_allclose(turnover,[.8,1.25])
    assert (cost>0).all()


def test_fractional_cvar_equals_ru_minimum_with_ties():
    losses = np.array([-7.,-2,0,0,1,4,4,4,8,13,21])
    for alpha in (.61,.95):
        objective = lambda eta: eta+np.maximum(losses-eta,0).mean()/(1-alpha)
        result = minimize_scalar(objective,bounds=(-10,25),method="bounded",options={"xatol":1e-10})
        _,cvar = empirical_cvar(losses,alpha)
        assert abs(result.fun-cvar)<1e-6


def test_policy_causality_and_pnl_gradient():
    torch.manual_seed(915)
    policy = PolicyMLP(8)
    model = Model()
    prices = torch.full((3,253),100.)
    altered = prices.clone(); altered[:,101:] = 500
    indices = np.arange(252)
    np.testing.assert_array_equal(neural_delta(policy,prices,model,indices)[:,:101].detach(),
                                  neural_delta(policy,altered,model,indices)[:,:101].detach())
    short = Model(T=2/252,steps=2)
    s = torch.tensor([[100.,101.,103.]],dtype=torch.float64)
    delta = torch.tensor([[.2,.4]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda d: hedging_pnl_torch(s,d,np.array([0,1]),.001,short), (delta,), eps=1e-6,atol=1e-6)


def test_table_and_bootstrap_share_exact_evaluation_losses():
    model = Model()
    paths = gbm_paths_from_z_np(model,np.random.default_rng(112).normal(size=(40,252)))
    policy = PolicyMLP(8).double()
    tensor = torch.tensor(paths)
    table = evaluate_neural(model,policy,tensor,paths)
    for freq in (1,5,21):
        for cost in (0.,.001,.005):
            losses = _neural_losses(model,policy,tensor,freq,cost,paths)
            assert empirical_cvar(losses)[1] == table[str(freq)][str(cost)]["cvar95"]
