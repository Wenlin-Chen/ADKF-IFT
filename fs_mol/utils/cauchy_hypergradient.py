from typing import Tuple
import torch


def cauchy_hypergradient(
    f_outer, 
    f_inner, 
    params_outer: Tuple[torch.Tensor], 
    params_inner: Tuple[torch.Tensor],
    device,
    ignore_grad_correction: bool = False, # ignores the second order term (only use the direct gradient)
    sanity_checks: bool = False,  # turn off in later versions to speed up
    ignore_direct_grad: bool = False, # for testing only, ignores the direct gradient df/d(outer)
):
    """
    Calculate the hypergradient of the f_outer function w.r.t.
    the outer parameters, where the inner parameters are at a value
    which (locally) minimizes f_inner
    
    NOTE: this function ASSUMES that you've already done the inner
    minimization somehow
    
    NOTE: Both inner and outer functions are to be called as
    f(params_outer, params_inner)
    """
    
    # 0: clean up existing gradients
    # (unsure what effect this might have,
    # but don't want to risk a problem)
    for tup in (params_outer, params_inner):
        for tensor in tup:
            tensor.grad = None

    if not ignore_grad_correction:
            
        # ==============
        # Get all values from f_inner
        # ==============
        
        # 1: find Hessian
        def _f_inner_only(*p_in): # note the * because torch expands the tuple to call f
            return f_inner(params_outer, p_in)
        hessian_tuple = torch.autograd.functional.hessian(
            _f_inner_only,
            params_inner,
        )
        
        # 2: Reshape it into a square matrix
        h_len = sum(t.nelement() for t in params_inner)
        H = torch.zeros((h_len, h_len)).to(device)  # this is the proper Hessian
        i_cum = 0
        for i in range(len(params_inner)):
            i_nelem = params_inner[i].nelement()

            j_cum = 0
            for j in range(len(params_inner)):
                j_nelem = params_inner[j].nelement()

                h_block = hessian_tuple[i][j].reshape(i_nelem, j_nelem)
                H[i_cum:i_cum + i_nelem, j_cum:j_cum + j_nelem] = h_block

                j_cum += j_nelem
            i_cum += i_nelem
        del i, j, i_cum, j_cum, i_nelem, j_nelem, h_block  # just in case
            
        # Sanity check: is the Hessian (approximately) invertible?
        # This check takes time, so should not be run in production
        if sanity_checks:
            logabsdet =  torch.linalg.slogdet(H).logabsdet
            if logabsdet < -2:
                print(
                    f"WARNING: determinant seems low ({logabsdet:.5g})."
                    " perhaps Hessian is not invertible?"
                )
            assert logabsdet.item() > -10.0
            
        # 3: find mixed partial derivatives
        def _f_inner_jac(p_out, p_in):
            return torch.autograd.functional.jacobian(
                lambda *_p_in_dummy: f_inner(p_out, _p_in_dummy),
                p_in,
                create_graph=True
            )
        mixed_partial_tuples = torch.autograd.functional.jacobian(
            lambda *_p_out_dummy: _f_inner_jac(_p_out_dummy, params_inner),
            params_outer
        )
        
        # 4: reshape it into a series of tensors
        # (one for each outer parameter)
        mixed_partials = []
        for j, p_o in enumerate(params_outer):
            
            J = torch.zeros((h_len, *tuple(p_o.shape))).to(device)
            
            i_cum = 0
            for i, p_i in enumerate(params_inner):
                i_nelem = p_i.nelement()
                J_block = mixed_partial_tuples[i][j].reshape(
                    i_nelem, *tuple(p_o.shape)
                )
                J[i_cum:i_cum + i_nelem] = J_block
                i_cum += i_nelem
            
            # Clean up
            mixed_partials.append(J)
            del i_cum, i_nelem, J, J_block
        
        # ==============
        # Run the actual forward pass
        # ==============
        
        # Check that there are no remaining gradients
        if sanity_checks:
            for tup in (params_outer, params_inner):
                for tensor in tup:
                    assert tensor.grad is None
                
    # 5: Run the forward pass through the outer function
    f_value = f_outer(params_outer, params_inner)
    f_value.backward()
    
    if not ignore_grad_correction:
        # ==============
        # All info is here: now calculate the gradient
        # ==============
        
        # 6: compute v := df/d(inner) * H^{-1}
        dfout_by_dpinner = torch.zeros(h_len).to(device)
        i_cum = 0
        for i, p_i in enumerate(params_inner):
            i_nelem = p_i.nelement()
            dfout_by_dpinner[i_cum:i_cum+i_nelem] = p_i.grad.flatten()
            i_cum += i_nelem
        del i_cum, i_nelem
        v = torch.linalg.solve(H, dfout_by_dpinner)
        
    # 7: *subtract* the correction term from the gradients
    for j, p_o in enumerate(params_outer):
        
        # Gradient might not exist if outer function
        # doesn't depend on an outer output
        # In that case, populate it
        if p_o.grad is None:
            p_o.grad = torch.zeros_like(p_o).to(device)
            p_o.grad.requires_grad_(False)
        
        # Optionally, get rid of direct gradient
        # For debugging only, this makes the calculation incorrect
        if ignore_direct_grad:
            p_o.grad.zero_()
            
        if not ignore_grad_correction:
            # apply df/d(inner) * H^{-1} to each gradient
            grad_correction = torch.tensordot(
                v, mixed_partials[j], dims=1  # dot prod over 1st dimension
            )
            
            # Add the correction
            assert p_o.grad.shape == grad_correction.shape
            p_o.grad -= grad_correction  # NOTE: the minus sign
    

    return f_value