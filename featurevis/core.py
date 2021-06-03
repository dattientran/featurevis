import warnings

import torch
from torch import optim
import numpy as np
from featurevis.exceptions import FeatureVisException


def gradient_ascent(f, x, transform=None, regularization=None, gradient_f=None,
                    post_update=None, optim_name='SGD', step_size=0.1, optim_kwargs={}, additional_kwargs={},
                    num_iterations=1000, save_iters=None, print_iters=100):
    """ Maximize f(x) via gradient ascent.

    Objective: f(transform(x)) - regularization(transform(x))
    Update: x_{t+1} = post_update(x_{t} + step_size * gradient_f(x_{t}.grad))

    Arguments:
        f (function): Real-valued differentiable function to be optimized
        x (torch.Tensor): Initial guess of the input to optimize.
        transform (function): Differentiable transformation applied to x before sending it
            through the model, e.g., an image generator, jittering, scaling, etc.
        regularization (function): Differentiable regularization term, e.g., natural
            prior, total variation, bilateral filters, etc.
        gradient_f (function): Non-differentiable. Receives the gradient of x and outputs
            a preconditioned gradient, e.g., blurring, masking, etc.
        post_update (function): Non-differentiable. Function applied to x after each
            gradient update, e.g., keep the image norm to some value, blurring, etc.
        optim_name (string): Optimizer to use: SGD or Adam.
        step_size (float): Size of the step size to give every iteration.
        optim_kwargs (dict): Dictionary with kwargs for the optimizer
        num_iterations (int): Number of gradient ascent steps.
        save_iters (None or int): How often to save x. If None, it returns the best x;
            otherwise it saves x after each save_iters iterations.
        print_iters (int): Print some results every print_iters iterations.

    Returns:
        optimal_x (torch.Tensor): x that maximizes the desired function. If save_iters is
            not None, this will be a list of tensors.
        fevals (list): Function evaluations at each iteration. We also evaluate at x_0
            (the original input) so this will have max_iterations + 1 elements.
        reg_terms (list): Value of the regularization term at each iteration. We also
            evaluate at x_0 (the original input) so this will have max_iterations + 1
            elements. Empty if regularization is None.

    Note:
        transform, regularization, gradient_f and post_update receive one positional
        parameter (its input) and the following optional named parameters:
            iteration (int): Current iteration (starts at 1).

        The number of optional parameters may increase so we recommend to write functions
        that receive **kwargs (or use the varargin decorator below) to make sure they will
        still work if we add other optional parameters in the future.
    """
    # Basic checks
    if x.dtype != torch.float32:
        raise ValueError('x must be of torch.float32 dtype')
    x = x.detach().clone()  # to avoid changing original
    x.requires_grad_()

    # Declare optimizer
    if optim_name == 'SGD':
        optimizer = optim.SGD([x], lr=step_size, **optim_kwargs)
    elif optim_name == 'Adam':
        optimizer = optim.Adam([x], lr=step_size, **optim_kwargs)
    else:
        raise ValueError("Expected optim_name to be 'SGD' or 'Adam'")

    # Run gradient ascent
    fevals = []  # to store function evaluations
    reg_terms = []  # to store regularization function evaluations
    saved_xs = []  # to store xs (ignored if save_iters is None)
    for i in range(1, num_iterations + 1):
        # Zero gradients
        if x.grad is not None:
            x.grad.zero_()

        # Transform input
        transformed_x = x if transform is None else transform(x, iteration=i)

        # f(x)
        feval = f(transformed_x)
        fevals.append(feval.item())
        
        if additional_kwargs:
            # Stop optimization when feval reaches target activation 
            if feval >= additional_kwargs['target_level'] * additional_kwargs['mei_activation']:
                break

        # Regularization
        if regularization is not None:
            reg_term = regularization(transformed_x, iteration=i)
            reg_terms.append(reg_term.item())
        else:
            reg_term = 0

        # Compute gradient
        (-feval + reg_term).backward()
        if x.grad is None:
            raise FeatureVisException('Gradient did not reach x.')

        # Precondition gradient
        x.grad = x.grad if gradient_f is None else gradient_f(x.grad, iteration=i)
        if (torch.abs(x.grad) < 1e-9).all():
            warnings.warn('Gradient for x is all zero')

        # Gradient ascent step (on x)
        optimizer.step()

        # Cleanup
        if post_update is not None:
            with torch.no_grad():
                x[:] = post_update(x, iteration=i)  # in place so the optimizer still points to the right object

        # Report results
        if i % print_iters == 0:
            feval = feval.item()
            reg_term = reg_term if regularization is None else reg_term.item()
            x_std = x.std().item()
            print('Iter {}: f(x) = {:.2f}, reg(x) = {:.2f}, std(x) = {:.2f}'.format(i,
                feval, reg_term, x_std))

        # Save x
        if save_iters is not None and i % save_iters == 0:
            saved_xs.append(x.detach().clone())

    # Record f(x) and regularization(x) for the final x
    with torch.no_grad():
        transformed_x = x if transform is None else transform(x, iteration=i + 1)

        feval = f(transformed_x)
        fevals.append(feval.item())

        if regularization is not None:
            reg_term = regularization(transformed_x, iteration=i + 1)
            reg_terms.append(reg_term.item())
    print('Final f(x) = {:.2f}'.format(fevals[-1]))

    # Set opt_x
    opt_x = x.detach().clone() if save_iters is None else saved_xs

    return opt_x, fevals, reg_terms

def param_gradient_ascent(f, x, mask_params, clipping=None, transform=None, texture_blur=None, regularization=None, gradient_f=None,
                    post_update=None, optim_name='SGD', step_size=0.1, optim_kwargs={}, additional_kwargs={},
                    num_iterations=1000, save_iters=None, print_iters=100):
    
    # Basic checks
    if x.dtype != torch.float32:
        raise ValueError('x must be of torch.float32 dtype')
    x = x.detach().clone()  # to avoid changing original
    x.requires_grad_()
    
    if mask_params.dtype != torch.float32:
        raise ValueError('mask_params must be of torch.float32 dtype')
    mask_params = mask_params.detach().clone()  # to avoid changing original
    mask_params.requires_grad_()

    # Declare optimizer
    if optim_name == 'SGD':
        optimizer = optim.SGD([x, mask_params], lr=step_size, **optim_kwargs)
    elif optim_name == 'Adam':
        optimizer = optim.Adam([x, mask_params], lr=step_size, **optim_kwargs)
    else:
        raise ValueError("Expected optim_name to be 'SGD' or 'Adam'")

    # Run gradient ascent
    fevals = []  # to store function evaluations
    reg_terms = []  # to store regularization function evaluations
    saved_xs = []  # to store xs (ignored if save_iters is None)
    for i in range(1, num_iterations + 1):
        # Zero gradients
        if x.grad is not None:
            x.grad.zero_()
        if mask_params.grad is not None:
            mask_params.grad.zero_()
        
        # Clip input
        clipped_params = mask_params if clipping is None else clipping(mask_params, iteration=i)
            
        # Transform input
        transformed_x = x if transform is None else transform(x, clipped_params, iteration=i)[0]
        mask_f = transform(x, clipped_params, iteration=i)[1]
        
        # f(x)
        feval = f(transformed_x)
        fevals.append(feval.item())
        
        if additional_kwargs:
            # Stop optimization when feval reaches target activation 
            if feval >= additional_kwargs['target_level'] * additional_kwargs['mei_activation']:
                break

        # Regularization
        if regularization is not None:
            if texture_blur is None:
                reg_term = regularization(transformed_x, iteration=i)
            else:
                reg_term = regularization(transform(texture_blur(x), clipped_params, iteration=i)[0], iteration=i)            
            reg_terms.append(reg_term.item())
        else:
            reg_term = 0

        # Compute gradient
        (-feval + reg_term).backward()
        if x.grad is None:
            raise FeatureVisException('Iter{}: Gradient did not reach x.'.format(i))
        if mask_params.grad is None:
            raise FeatureVisException('Iter{}: Gradient did not reach mask_params.'.format(i))

        # Precondition gradient
        x.grad = x.grad if gradient_f is None else gradient_f(x.grad, iteration=i)
        if (torch.abs(x.grad) < 1e-9).all():
            warnings.warn('Iter{}: Gradient for x is all zero'.format(i))
        if (torch.abs(mask_params.grad) < 1e-9).all():
            warnings.warn('Iter{}: Gradient for mask_params is all zero'.format(i))
            
        # Gradient ascent step (on x)
        optimizer.step()

        # Cleanup
        if post_update is not None:
            with torch.no_grad():
                x[:] = post_update(x, iteration=i)  # in place so the optimizer still points to the right object

        # Report results
        if i % print_iters == 0:
            feval = feval.item()
            reg_term = reg_term if regularization is None else reg_term.item()
            x_std = x.std().item()
            print('Iter {}: f(x) = {:.2f}, reg(x) = {:.2f}, std(x) = {:.2f}'.format(i,
                feval, reg_term, x_std))

        # Save x
        if save_iters is not None and i % save_iters == 0:
            saved_xs.append(x.detach().clone())

    # Record f(x) and regularization(x) for the final x
    with torch.no_grad():
        clipped_params = mask_params if clipping is None else clipping(mask_params, iteration=i + 1)
        transformed_x = x if transform is None else transform(x, clipped_params, iteration=i + 1)[0]
        mask_f = transform(x, clipped_params, iteration=i + 1)[1]
        
        feval = f(transformed_x)
        fevals.append(feval.item())

        if regularization is not None:
            if texture_blur is None:
                reg_term = regularization(transformed_x, iteration=i+1)
            else:
                reg_term = regularization(transform(texture_blur(x), clipped_params, iteration=i+1)[0], iteration=i+1)
            reg_terms.append(reg_term.item())
    print('Final f(x) = {:.2f}'.format(fevals[-1]))

    # Set opt_x
    opt_x = x.detach().clone() if save_iters is None else saved_xs
    opt_params = clipped_params.detach().clone()
    mask_f = mask_f.detach().clone()

    return opt_x, opt_params, mask_f, fevals, reg_terms

def px_gradient_ascent(f, x_f, mask_v, texture, transform=None, texture_blur=None, regularization=None, text_postup=None, 
                      image_postup=None, gradient_f=None, optim_name='SGD', step_size=0.1, optim_kwargs={}, additional_kwargs={},
                    num_iterations=1000, print_iters=100):
    
    # Basic checks
    if x_f.dtype != torch.float32:
        raise ValueError('x_f must be of torch.float32 dtype')
    x_f = x_f.detach().clone()  # to avoid changing original
    x_f.requires_grad_()
    
    if mask_v.dtype != torch.float32:
        raise ValueError('mask_v must be of torch.float32 dtype')
    mask_v = mask_v.detach().clone()  # to avoid changing original
    mask_v.requires_grad_()
    
    if texture.dtype != torch.float32:
        raise ValueError('texture must be of torch.float32 dtype')
    texture = texture.detach().clone()  # to avoid changing original
    texture.requires_grad_()

    # Declare optimizer
    if optim_name == 'SGD':
        optimizer = optim.SGD([x_f, mask_v, texture], lr=step_size, **optim_kwargs)
    elif optim_name == 'Adam':
        optimizer = optim.Adam([x_f, mask_v, texture], lr=step_size, **optim_kwargs)
    else:
        raise ValueError("Expected optim_name to be 'SGD' or 'Adam'")

    # Run gradient ascent
    fevals = []  # to store function evaluations
    reg_terms = []  # to store regularization function evaluations
    saved_xs = []  # to store xs (ignored if save_iters is None)
    for i in range(1, num_iterations + 1):
        # Zero gradients
        if x_f.grad is not None:
            x_f.grad.zero_()
        if mask_v.grad is not None:
            mask_v.grad.zero_()
        if texture.grad is not None:
            texture.grad.zero_()
                    
        # Transform input
        transformed_x = x_f if transform is None else transform(x_f, mask_v, texture, iteration=i)
        
        # f(x)
        feval = f(transformed_x)
        fevals.append(feval.item())
        
        if additional_kwargs:
            # Stop optimization when feval reaches target activation 
            if feval >= additional_kwargs['target_level'] * additional_kwargs['mei_activation']:
                break

        # Regularization
        if regularization is not None:
            if texture_blur is None:
                reg_term = regularization(transformed_x, iteration=i)
            else:
                reg_term = regularization(transform(x_f, mask_v, texture_blur(texture), iteration=i), iteration=i)            
            reg_terms.append(reg_term.item())
        else:
            reg_term = 0

        # Compute gradient
        (-feval + reg_term).backward()
        if x_f.grad is None:
            raise FeatureVisException('Iter{}: Gradient did not reach x_f.'.format(i))
        # Precondition gradient
        x_f.grad = x_f.grad if gradient_f is None else gradient_f(x_f.grad, iteration=i)
        if (torch.abs(x_f.grad) < 1e-9).all():
            warnings.warn('Iter{}: Gradient for im is all zero'.format(i))
            
        if mask_v.grad is None:
            raise FeatureVisException('Iter{}: Gradient did not reach mask_v.'.format(i))
        # Precondition gradient
        mask_v.grad = mask_v.grad if gradient_f is None else gradient_f(mask_v.grad, iteration=i)
        if (torch.abs(mask_v.grad) < 1e-9).all():
            warnings.warn('Iter{}: Gradient for im is all zero'.format(i))
            
        if texture.grad is None:
            raise FeatureVisException('Iter{}: Gradient did not reach texture.'.format(i))
        # Precondition gradient
        texture.grad = texture.grad if gradient_f is None else gradient_f(texture.grad, iteration=i)
        if (torch.abs(texture.grad) < 1e-9).all():
            warnings.warn('Iter{}: Gradient for im is all zero'.format(i))
    
        # Gradient ascent step (on x)
        optimizer.step()

        # Cleanup
        if text_postup is not None:
            with torch.no_grad():
                texture[:] = text_postup(texture, iteration=i)
        if image_postup is not None:
            with torch.no_grad():
                x_f[:] = image_postup(x_f, iteration=i)

        # Report results
        if i % print_iters == 0:
            feval = feval.item()
            reg_term = reg_term if regularization is None else reg_term.item()
            x_std = transformed_x.std().item()
            print('Iter {}: f(x) = {:.2f}, reg(x) = {:.2f}, std(x) = {:.2f}'.format(i,
                feval, reg_term, x_std))

    # Record f(x) and regularization(x) for the final x
    with torch.no_grad():
        transformed_x = x_f if transform is None else transform(x_f, mask_v, texture, iteration=i)        
        feval = f(transformed_x)
        fevals.append(feval.item())

        if regularization is not None:
            if texture_blur is None:
                reg_term = regularization(transformed_x, iteration=i+1)
            else:
                reg_term = regularization(transform(x_f, mask_v, texture_blur(texture), iteration=i+1), iteration=i+1)
            reg_terms.append(reg_term.item())
    print('Final f(x) = {:.2f}'.format(fevals[-1]))
                      
    # Save final optimized images          
    opt_x_f = x_f.detach().clone()
    opt_mask_v = mask_v.detach().clone()
    opt_texture = texture.detach().clone()
                      
    return opt_x_f, opt_mask_v, opt_texture, fevals, reg_terms
    

def contour_walk(f, f_act, x, mei_act, random_dir=None, target_level=0.85, dev_thre=0, seed=0, transform=None, regularization=None, gradient_f=None,
                    post_update=None, optim_name='SGD', step_size=0.1, optim_kwargs={}, 
                    num_iterations=1000, save_iters=None, print_iters=100):
    # Basic checks
    if x.dtype != torch.float32:
        raise ValueError('x must be of torch.float32 dtype')
    x = x.detach().clone()  # to avoid changing original
    x.requires_grad_()

    # Declare optimizer
    if optim_name == 'SGD':
        optimizer = optim.SGD([x], lr=step_size, **optim_kwargs)
    elif optim_name == 'Adam':
        optimizer = optim.Adam([x], lr=step_size, **optim_kwargs)
    else:
        raise ValueError("Expected optim_name to be 'SGD' or 'Adam'")

    # Run gradient ascent
    fevals = []  # to store function evaluations
    reg_terms = []  # to store regularization function evaluations
    activations = []
    saved_xs = [x.detach().clone()]  # to store xs (ignored if save_iters is None)
    
    torch.manual_seed(seed)

    for i in range(1, num_iterations + 1):
        # keep walking until walks off the equal activation contour with small amount of allowed deviation:            
        # Zero gradients
        if x.grad is not None:
            x.grad.zero_()

        # Transform input
        transformed_x = x if transform is None else transform(x, iteration=i)

        # f(x)
        feval = f(transformed_x)
        fevals.append(feval.item())
        
        # get image activation
        act = f_act(transformed_x).item()
        activations.append(act)
        
        # Regularization
        if regularization is not None:
            reg_term = regularization(transformed_x, iteration=i)
            reg_terms.append(reg_term.item())
        else:
            reg_term = 0
            
        # Random walk or optimize back to the target activation level
        if random_dir is None:  # keep optimizing back to target activation level
            (-feval + reg_term).backward()
            if gradient_f is not None:
                x.grad = gradient_f(x.grad, iteration=i)
            x.grad = x.grad / torch.norm(x.grad)
            
        elif (act >= mei_act * (target_level - dev_thre)):# and (act <= mei_act * (target_level + dev_thre)):
            direction = torch.randn(x.shape).cuda()
            if random_dir == 'random':  # random walk
                x.grad = direction

            elif random_dir == 'ortho':  # walk orthogonal to gradient direction
                f_act(transformed_x).backward()
                grad_norm = torch.sqrt(torch.sum(x.grad**2)) 
                walk_direction = direction - (torch.dot(direction.view(-1), x.grad.view(-1)) / grad_norm**2) * x.grad
                x.grad = walk_direction
                
            if gradient_f is not None:
                x.grad = gradient_f(x.grad, iteration=i)
            x.grad = x.grad / torch.norm(x.grad)
            
        else:
            (-feval + reg_term).backward()
            if gradient_f is not None:
                x.grad = gradient_f(x.grad, iteration=i)
            x.grad = x.grad / torch.norm(x.grad)

        if (torch.abs(x.grad) < 1e-9).all():
            warnings.warn('Gradient for x is all zero')

        # Gradient ascent step (on x)
        optimizer.step()

        # Cleanup
        if post_update is not None:
            with torch.no_grad():
                x[:] = post_update(x, iteration=i)  # in place so the optimizer still points to the right object

        # Report results
        if i % print_iters == 0:
            feval = feval.item()
            reg_term = reg_term if regularization is None else reg_term.item()
            x_std = x.std().item()
            print('Iter {}: f(x) = {:.2f}, reg(x) = {:.2f}, std(x) = {:.2f}'.format(i,
                feval, reg_term, x_std))

        # Save x
        if save_iters is not None and i % save_iters == 0:
            saved_xs.append(x.detach().clone())

            
    # Record f(x) and regularization(x) for the final x
    with torch.no_grad():
        transformed_x = x if transform is None else transform(x, iteration=i + 1)

        feval = f(transformed_x)
        fevals.append(feval.item())

        if regularization is not None:
            reg_term = regularization(transformed_x, iteration=i + 1)
            reg_terms.append(reg_term.item())
    print('Final f(x) = {:.2f}'.format(fevals[-1]))

    # Set opt_x
    opt_x = x.detach().clone() if save_iters is None else saved_xs

    return opt_x, activations, fevals, reg_terms


def save_intermediate_gradient_ascent(f, x, target_f, transform=None, regularization=None, gradient_f=None,
                    post_update=None, optim_name='SGD', step_size=1, optim_kwargs={},
                    num_iterations=1000, save_levels=np.linspace(0, 1, 11), print_iters=100):
    """ Maximize f(x) via gradient ascent.
    Objective: f(transform(x)) - regularization(transform(x))
    Update: x_{t+1} = post_update(x_{t} + step_size * gradient_f(x_{t}.grad))
    Arguments:
        f (function): Real-valued differentiable function to be optimized
        x (torch.Tensor): Initial guess of the input to optimize.
        transform (function): Differentiable transformation applied to x before sending it
            through the model, e.g., an image generator, jittering, scaling, etc.
        regularization (function): Differentiable regularization term, e.g., natural
            prior, total variation, bilateral filters, etc.
        gradient_f (function): Non-differentiable. Receives the gradient of x and outputs
            a preconditioned gradient, e.g., blurring, masking, etc.
        post_update (function): Non-differentiable. Function applied to x after each
            gradient update, e.g., keep the image norm to some value, blurring, etc.
        optim_name (string): Optimizer to use: SGD or Adam.
        step_size (float): Size of the step size to give every iteration.
        optim_kwargs (dict): Dictionary with kwargs for the optimizer
        num_iterations (int): Number of gradient ascent steps.
        save_levels (None or list of activation levels): At what activation level to save x. If None, it returns the best x;
            otherwise it saves x the first time the activation reaches a certain level
        target_f: the target optimized f value
        print_iters (int): Print some results every print_iters iterations.
    Returns:
        optimal_x (torch.Tensor): x that maximizes the desired function. If save_iters is
            not None, this will be a list of tensors.
        fevals (list): Function evaluations at each iteration. We also evaluate at x_0
            (the original input) so this will have max_iterations + 1 elements.
        reg_terms (list): Value of the regularization term at each iteration. We also
            evaluate at x_0 (the original input) so this will have max_iterations + 1
            elements. Empty if regularization is None.
        grads (list): Value of gradients at each iteration
        
    Note:
        transform, regularization, gradient_f and post_update receive one positional
        parameter (its input) and the following optional named parameters:
            iteration (int): Current iteration (starts at 1).
        The number of optional parameters may increase so we recommend to write functions
        that receive **kwargs (or use the varargin decorator below) to make sure they will
        still work if we add other optional parameters in the future.
    """
    
    fevals = []  # to store function evaluations
    reg_terms = []  # to store regularization function evaluations
    saved_xs = []  # to store the first xs that is optimized above certain activation levels (ignored if save_iters is None)
    acts = [] # to store activation of saved_xs
    level = save_levels[0]
    level_count = 1
    level_iters = 1
    
    # Basic checks
    if x.dtype != torch.float32:
        raise ValueError('x must be of torch.float32 dtype')
    x = x.detach().clone()  # to avoid changing original
        
    # save initial x
    if save_levels is not None:
        saved_xs.append(x.detach().clone())
        acts.append(f(x).item())

    x.requires_grad_()

    # Declare optimizer
    if optim_name == 'SGD':
        optimizer = optim.SGD([x], lr=step_size, **optim_kwargs)
    elif optim_name == 'Adam':
        optimizer = optim.Adam([x], lr=step_size, **optim_kwargs)
    else:
        raise ValueError("Expected optim_name to be 'SGD' or 'Adam'")

    # Run gradient ascent 
    for i in range(1, num_iterations + 1):
        # Zero gradients
        if x.grad is not None:
            x.grad.zero_()

        # Transform input
        transformed_x = x if transform is None else transform(x, iteration=i)
            
        # f(x)
        feval = f(transformed_x)
        fevals.append(feval.item())

        # Regularization
        if regularization is not None:
            reg_term = regularization(transformed_x, iteration=i)
            reg_terms.append(reg_term.item())
        else:
            reg_term = 0

        # Compute gradient
        (-feval + reg_term).backward()
        if x.grad is None:
            raise FeatureVisException('Gradient did not reach x.')

        # Precondition gradient
        x.grad = x.grad if gradient_f is None else gradient_f(x.grad, iteration=i)
        if (torch.abs(x.grad) < 1e-9).all():
            warnings.warn('Gradient for x is all zero')
        
        # Gradient ascent step (on x)
        optimizer.step()

        # Cleanup
        if post_update is not None:
            with torch.no_grad():
                x[:] = post_update(x, iteration=i)  # in place so the optimizer still points to the right object

        # Report results
        if i % print_iters == 0:
            feval = feval.item()
            reg_term = reg_term if regularization is None else reg_term.item()
            x_std = x.std().item()
            print('Iter {}: f(x) = {:.2f}, reg(x) = {:.2f}, std(x) = {:.2f}'.format(i,
                feval, reg_term, x_std))
        
        # Save x
        if save_levels is not None and level_count <= len(save_levels) and f(x).item() >= level * target_f:   # f(x) is the activation at the current iter, feval is the previous iter
            
            if i != 1 and fevals[level_iters] >= level* target_f:
                saved_xs.append(saved_xs[-1])
                acts.append(acts[-1])
                
            else:
                saved_xs.append(x.detach().clone())
                level_iters = i 
                acts.append(f(x).item())
                
#             if i == 1 or fevals[level_iters] <= level* target_f:
#                 saved_xs.append(x.detach().clone())
#                 acts.append(f(x).item())
#                 level_iters = i 
                
            level_count += 1
            if level_count <= len(save_levels):
                level = save_levels[level_count-1]
            
    # Record f(x) and regularization(x) for the final x
    with torch.no_grad():
        transformed_x = x if transform is None else transform(x, iteration=i + 1)

        feval = f(transformed_x)
        fevals.append(feval.item())

        if regularization is not None:
            reg_term = regularization(transformed_x, iteration=i + 1)
            reg_terms.append(reg_term.item())
            
            
    print('Final f(x) = {:.2f}'.format(fevals[-1]))

    # Set opt_x
    saved_xs.append(x.detach().clone())
    acts.append(f(x).item())
    opt_x = x.detach().clone() if save_levels is None else saved_xs
    
    return opt_x, acts, fevals, reg_terms