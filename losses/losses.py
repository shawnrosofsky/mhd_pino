import numpy as np
import torch
import torch.nn.functional as F
import math

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class LossMHD(object):
    def __init__(self, nu=1e-4, eta=1e-4, rho0=1.0, 
                 data_weight=1.0, ic_weight=1.0, pde_weight=1.0, constraint_weight=1.0,
                 use_data_loss=True, use_ic_loss=True, use_pde_loss=True, use_constraint_loss=True,
                 u_weight=1.0, v_weight=1.0, Bx_weight=1.0, By_weight=1.0, 
                 Du_weight=1.0, Dv_weight=1.0, DBx_weight=1.0, DBy_weight=1.0, 
                 div_B_weight=1.0, div_vel_weight=1.0, 
                 Lx=1.0, Ly=1.0, tend=1.0, 
                 device=None):
        self.nu = nu
        self.eta = eta
        self.rho0 = rho0
        self.data_weight = data_weight
        self.ic_weight = ic_weight
        self.pde_weight = pde_weight
        self.constraint_weight = constraint_weight
        self.use_data_loss = use_data_loss
        self.use_ic_loss = use_ic_loss
        self.use_pde_loss = use_pde_loss
        self.use_constraint_loss = use_constraint_loss
        self.u_weight = u_weight
        self.v_weight = v_weight
        self.Bx_weight = Bx_weight
        self.By_weight = By_weight
        self.Du_weight = Du_weight
        self.Dv_weight = Dv_weight
        self.DBx_weight = DBx_weight
        self.DBy_weight = DBy_weight
        self.div_B_weight = div_B_weight
        self.div_vel_weight = div_vel_weight
        self.Lx = Lx
        self.Ly = Ly
        self.tend = tend
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
    
    def __call__(self, pred, true, inputs, return_loss_dict=False):
        if not return_loss_dict:
            loss = self.compute_loss(pred, true, inputs)
            return loss
        else:
            loss, loss_dict = self.compute_losses(pred, true, inputs)
            return loss, loss_dict
        
    
    def compute_loss(self, pred, true, inputs):
        pred = pred.reshape(true.shape)
        u = pred[..., 0]
        v = pred[..., 1]
        Bx = pred[..., 2]
        By = pred[..., 3]
        
        # Data
        if self.use_data_loss:
            loss_data = self.data_loss(pred, true)
        else:
            loss_data = 0
        # IC
        if self.use_ic_loss:
            loss_ic = self.ic_loss(pred, inputs)
        else:
            loss_ic = 0
            
        # PDE
        if self.use_pde_loss:
            Du, Dv, DBx, DBy = self.mhd_pde(u, v, Bx, By)
            loss_pde = self.mhd_pde_loss(Du, Dv, DBx, DBy)
        else:
            loss_pde = 0
        
        # Constraints
        if self.use_constraint_loss:
            div_vel, div_B = self.mhd_constraint(self, u, v, Bx, By)
            loss_constraints = self.mhd_constraint_loss(div_vel, div_B)
        else:
            loss_constraints = 0

        loss = self.data_weight*loss_data + self.ic_weight*loss_ic + self.pde_weight*loss_pde + self.constraint_weight*loss_constraints
        return loss
    
    def compute_losses(self, pred, true, inputs):
        pred = pred.reshape(true.shape)
        u = pred[..., 0]
        v = pred[..., 1]
        Bx = pred[..., 2]
        By = pred[..., 3]
        
        loss_dict = {}
        
        # Data
        if self.use_data_loss:
            loss_data, loss_u, loss_v, loss_Bx, loss_By = self.data_loss(pred, true, return_all_losses=True)
            loss_dict['loss_data'] = loss_data
            loss_dict['loss_u'] = loss_u
            loss_dict['loss_v'] = loss_v
            loss_dict['loss_Bx'] = loss_Bx
            loss_dict['loss_By'] = loss_By
        else:
            loss_data = 0
        # IC
        if self.use_ic_loss:
            loss_ic, loss_u_ic, loss_v_ic, loss_Bx_ic, loss_By_ic = self.ic_loss(pred, inputs, return_all_losses=True)
            loss_dict['loss_ic'] = loss_ic
            loss_dict['loss_u_ic'] = loss_u_ic
            loss_dict['loss_v_ic'] = loss_v_ic
            loss_dict['loss_Bx_ic'] = loss_Bx_ic
            loss_dict['loss_By_ic'] = loss_By_ic
        else:
            loss_ic = 0
            
        # PDE
        if self.use_pde_loss:
            Du, Dv, DBx, DBy = self.mhd_pde(u, v, Bx, By)
            loss_pde, loss_Du, loss_Dv, loss_DBx, loss_DBy = self.mhd_pde_loss(Du, Dv, DBx, DBy, return_all_losses=True)
            loss_dict['loss_pde'] = loss_pde
            loss_dict['loss_Du'] = loss_Du
            loss_dict['loss_Dv'] = loss_Dv
            loss_dict['loss_DBx'] = loss_DBx
            loss_dict['loss_DBy'] = loss_DBy
        else:
            loss_pde = 0
        
        # Constraints
        if self.use_constraint_loss:
            div_vel, div_B = self.mhd_constraint(self, u, v, Bx, By)
            loss_constraint, loss_div_vel, loss_div_B = self.mhd_constraint_loss(div_vel, div_B, return_all_losses=True)
            loss_dict['loss_constraint'] = loss_constraint
            loss_dict['loss_div_vel'] = loss_div_vel
            loss_dict['loss_div_B'] = loss_div_B
        else:
            loss_constraints = 0

        loss = self.data_weight*loss_data + self.ic_weight*loss_ic + self.pde_weight*loss_pde + self.constraint_weight*loss_constraints
        loss_dict['loss'] = loss
        return loss, loss_dict

    
    def data_loss(self, pred, true, return_all_losses=False):
        lploss = LpLoss(size_average=True)
        u_pred = pred[..., 0]
        v_pred = pred[..., 1]
        Bx_pred = pred[..., 2]
        By_pred = pred[..., 3]
        
        u_true = true[..., 0]
        v_true = true[..., 1]
        Bx_true = true[..., 2]
        By_true = true[..., 3]
        
        loss_u = lploss(u_pred, u_true)
        loss_v = lploss(v_pred, v_true)
        loss_Bx = lploss(Bx_pred, Bx_true)
        loss_By = lploss(By_pred, By_true)
        
        loss_data = self.u_weight*loss_u + self.v_weight*loss_v + self.Bx_weight*loss_Bx + self.By_weight*loss_By
        if return_all_losses:
            return loss_data, loss_u, loss_v, loss_Bx, loss_By
        else:
            return loss_data
    
    def ic_loss(self, pred, inputs, return_all_losses=False):
        lploss = LpLoss(size_average=True)
        ic_pred = pred[:, 0]
        ic_true = inputs[:, 0, ..., 3:]
        u_ic_pred = ic_pred[..., 0]
        v_ic_pred = ic_pred[..., 1]
        Bx_ic_pred = ic_pred[..., 2]
        By_ic_pred = ic_pred[..., 3]
        
        u_ic_true = ic_true[..., 0]
        v_ic_true = ic_true[..., 1]
        Bx_ic_true = ic_true[..., 2]
        By_ic_true = ic_true[..., 3]
        
        loss_u_ic = lploss(u_ic_pred, u_ic_true)
        loss_v_ic = lploss(v_ic_pred, v_ic_true)
        loss_Bx_ic = lploss(Bx_ic_pred, Bx_ic_true)
        loss_By_ic = lploss(By_ic_pred, By_ic_true)
        
        loss_ic = self.u_weight*loss_u_ic + self.v_weight*loss_v_ic + self.Bx_weight*loss_Bx_ic + self.By_weight*loss_By_ic
        if return_all_losses:
            return loss_ic, loss_u_ic, loss_v_ic, loss_Bx_ic, loss_By_ic
        else:
            return loss_ic
    
    def mhd_pde_loss(self, Du, Dv, DBx, DBy, return_all_losses=None):
        Du_val = torch.zeros_like(Du)
        Dv_val = torch.zeros_like(Dv)
        DBx_val = torch.zeros_like(DBx)
        DBy_val = torch.zeros_like(DBy)

        loss_Du = F.mse_loss(Du, Du_val)
        loss_Dv = F.mse_loss(Dv, Dv_val)
        loss_DBx = F.mse_loss(DBx, DBx_val)
        loss_DBy = F.mse_loss(DBy, DBy_val)
        
        loss_pde = self.Du_weight*loss_Du + self.Dv_weight*loss_Dv + self.DBx_weight*loss_DBx + self.DBy_weight*loss_DBy
        if return_all_losses:
            return loss_pde, loss_Du, loss_Dv, loss_DBx, loss_DBy
        else:
            return loss_pde
    
    def mhd_constraint_loss(self, div_vel, div_B, return_all_losses=False):       
        div_vel_val = torch.zeros_like(div_vel)
        div_B_val = torch.zeros_like(div_B)
        
        loss_div_vel = F.mse_loss(div_vel, div_vel_val)
        loss_div_B = F.mse_loss(div_B, div_B_val)
        
        loss_constraint = self.div_vel_weight*loss_div_vel + self.div_B_weight*loss_div_B
        if return_all_losses:
            return loss_constraint, loss_div_vel, loss_div_B
        else:
            return loss_constraint
        

    def mhd_constraint(self, u, v, Bx, By):
        batchsize = u.size(0)
        nt = u.size(1)
        nx = u.size(2)
        ny = u.size(3)
        device = u.device
        dt = self.tend / (nt - 1)
        dx = self.Lx / nx
        dy = self.Ly / ny
        k_max = nx//2
        k_x = 2*np.pi/self.Lx * torch.cat([torch.arange(start=0, end=k_max, step=1, device=device),
                                           torch.arange(start=-k_max, end=0, step=1, device=device)], 0).reshape(nx, 1).repeat(1, ny).reshape(1,1,nx,ny)
        k_y = 2*np.pi/self.Ly * torch.cat([torch.arange(start=0, end=k_max, step=1, device=device),
                                           torch.arange(start=-k_max, end=0, step=1, device=device)], 0).reshape(1, ny).repeat(nx, 1).reshape(1,1,nx,ny)
        
        u_h = torch.fft.fftn(u, dim=[2, 3])
        v_h = torch.fft.fftn(v, dim=[2, 3])
        Bx_h = torch.fft.fftn(Bx, dim=[2, 3])
        By_h = torch.fft.fftn(By, dim=[2, 3])
        
        ux_h = self.Du_i(u_h, k_x)
        vy_h = self.Du_i(v_h, k_y)
        Bx_x_h = self.Du_i(Bx_h, k_x)
        By_y_h = self.Du_i(By_h, k_y)
        
        ux = torch.irfftn(ux_h[..., :k_max+1], dim=[2, 3])
        vy = torch.irfftn(vy_h[..., :k_max+1], dim=[2, 3])
        Bx_x = torch.irfftn(Bx_x_h[..., :k_max+1], dim=[2, 3])
        By_y = torch.irfftn(By_y_h[..., :k_max+1], dim=[2, 3])
        
        div_vel = ux + vy
        div_B = Bx_x + By_y
        return div_vel, div_B
    
    
        
    def mhd_pde(self, u, v, Bx, By):
        batchsize = u.size(0)
        nt = u.size(1)
        nx = u.size(2)
        ny = u.size(3)
        device = u.device
        dt = self.tend / (nt - 1)
        dx = self.Lx / nx
        dy = self.Ly / ny
        k_max = nx//2
        k_x = 2*np.pi/self.Lx * torch.cat([torch.arange(start=0, end=k_max, step=1, device=device),
                                           torch.arange(start=-k_max, end=0, step=1, device=device)], 0).reshape(nx, 1).repeat(1, ny).reshape(1,1,nx,ny)
        k_y = 2*np.pi/self.Ly * torch.cat([torch.arange(start=0, end=k_max, step=1, device=device),
                                           torch.arange(start=-k_max, end=0, step=1, device=device)], 0).reshape(1, ny).repeat(nx, 1).reshape(1,1,nx,ny)
        
        B2 = Bx**2 + By**2
        
        u_h = torch.fft.fftn(u, dim=[2, 3])
        v_h = torch.fft.fftn(v, dim=[2, 3])
        Bx_h = torch.fft.fftn(Bx, dim=[2, 3])
        By_h = torch.fft.fftn(By, dim=[2, 3])
        B2_h = torch.fft.fftn(B2, dim=[2, 3])
        
        ux_h = self.Du_i(u_h, k_x)
        uy_h = self.Du_i(u_h, k_y)
        vx_h = self.Du_i(v_h, k_x)
        vy_h = self.Du_i(v_h, k_y)
        
        Bx_x_h = self.Du_i(Bx_h, k_x)
        Bx_y_h = self.Du_i(Bx_h, k_y)
        By_x_h = self.Du_i(By_h, k_x)
        By_y_h = self.Du_i(By_h, k_y)
        
        u_lap_h = self.Lap(u_h, k_x, k_y)
        v_lap_h = self.Lap(v_h, k_x, k_y)
        Bx_lap_h = self.Lap(Bx_h, k_x, k_y)
        By_lap_h = self.Lap(By_h, k_x, k_y)
        
        # note that for pressure, the zero mode (the mean) cannot be zero for invertability so it is set to 1
        lap = -(k_x**2 + k_y**2)
        lap[0, 0, 0, 0] = -1.0

        p_h = -self.rho0*(ux_h**2 + 2*uy_h*vx_h + vy_h**2)/lap
        ptot_h = p_h + B2_h/2.0
        ptot_x_h = self.Du_i(ptot_h, k_x)
        ptot_y_h = self.Du_i(ptot_h, k_y)
        
        ux = torch.irfftn(ux_h[..., :k_max+1], dim=[2, 3])
        uy = torch.irfftn(uy_h[..., :k_max+1], dim=[2, 3])
        vx = torch.irfftn(vx_h[..., :k_max+1], dim=[2, 3])
        vy = torch.irfftn(vy_h[..., :k_max+1], dim=[2, 3])
        Bx_x = torch.irfftn(Bx_x_h[..., :k_max+1], dim=[2, 3])
        Bx_y = torch.irfftn(Bx_y_h[..., :k_max+1], dim=[2, 3])
        By_x = torch.irfftn(By_x_h[..., :k_max+1], dim=[2, 3])
        By_y = torch.irfftn(By_y_h[..., :k_max+1], dim=[2, 3])
        ptot = torch.irfftn(ptot_h[..., :k_max+1], dim=[2, 3])
        ptot_x = torch.irfftn(ptot_x_h[..., :k_max+1], dim=[2, 3])
        ptot_y = torch.irfftn(ptot_y_h[..., :k_max+1], dim=[2, 3])
        u_lap = torch.irfftn(u_lap_h[..., :k_max+1], dim=[2, 3])
        v_lap = torch.irfftn(v_lap_h[..., :k_max+1], dim=[2, 3])
        Bx_lap = torch.irfftn(Bx_lap_h[..., :k_max+1], dim=[2, 3])
        By_lap = torch.irfftn(By_lap_h[..., :k_max+1], dim=[2, 3])
        
        vel_grad_u = u*ux + v*uy
        vel_grad_v = u*vx + v*vy
        
        B_grad_u = Bx*ux + By*uy
        B_grad_v = Bx*vx + By*vy
        
        vel_grad_Bx = u*Bx_x + v*Bx_y
        vel_grad_By = u*By_x + v*By_y
        
        B_grad_Bx = Bx*Bx_x + By*Bx_y 
        B_grad_By = Bx*By_x + By*By_y 
        
        u_rhs = -vel_grad_u - ptot_x/self.rho0 + B_grad_Bx/self.rho0 + self.nu*u_lap
        v_rhs = -vel_grad_v - ptot_y/self.rho0 + B_grad_By/self.rho0 + self.nu*v_lap
        Bx_rhs = B_grad_u - vel_grad_Bx + self.eta*Bx_lap
        By_rhs = B_grad_v - vel_grad_By + self.eta*By_lap
        
        u_t = self.Du_t(u, dt)
        v_t = self.Du_t(v, dt)
        Bx_t = self.Du_t(Bx, dt)
        By_t = self.Du_t(By, dt)
        
        Du = u_t - u_rhs[:, 1:-1]
        Dv = v_t - v_rhs[:, 1:-1]
        DBx = Bx_t - Bx_rhs[:, 1:-1]
        DBy = By_t - By_rhs[:, 1:-1]
        
        return Du, Dv, DBx, DBy
    
    def Du_t(self, u, dt):
        u_t = (u[:, 2:] - u[:, :-2]) / (2*dt)
        return u_t
        
    def Lap(self, u_h, k_i, k_j):
        lap = -(k_i**2 + k_j**2)
        u_lap_h = lap * u_h
        return u_lap_h
    
    def Du_i(self, u_h, k_i):
        u_i_h = (1j*k_i) * u_h
        return u_i_h
    
    def Du_ij(self, u_h, k_i, k_j):
        u_ij_h = (1j*k_i) * (1j*k_j) * u_h
        return u_ij_h
    
    def Du_ii(self, u_h, k_i):
        u_ii_h = self.Du_ij(u_h, k_i, k_i)
        return u_ii_h
    
    
class LossMHDVecPot(LossMHD):
    def __init__(self, nu=1e-4, eta=1e-4, rho0=1.0, 
                 data_weight=1.0, ic_weight=1.0, pde_weight=1.0, constraint_weight=1.0,
                 use_data_loss=True, use_ic_loss=True, use_pde_loss=True, use_constraint_loss=True,
                 u_weight=1.0, v_weight=1.0, A_weight=1.0, 
                 Du_weight=1.0, Dv_weight=1.0, DA_weight=1.0, 
                 div_B_weight=1.0, div_vel_weight=1.0, 
                 Lx=1.0, Ly=1.0, tend=1.0, 
                 device=None):
        
        super().__init__(nu=nu, eta=eta, rho0=rho0,
                         data_weight=data_weight, ic_weight=ic_weight, pde_weight=pde_weight, constraint_weight=constraint_weight,
                         use_data_loss=use_data_loss, use_ic_loss=use_ic_loss, use_pde_loss=use_pde_loss, use_constraint_loss=use_constraint_loss,
                         u_weight=u_weight, v_weight=v_weight, 
                         Du_weight=Du_weight, Dv_weight=Dv_weight,
                         div_B_weight=div_B_weight, div_vel_weight=div_vel_weight, 
                         Lx=Lx, Ly=Ly, tend=tend, 
                         device=device)
        self.A_weight = A_weight
        self.DA_weight = DA_weight
    
    def compute_loss(self, pred, true, inputs):
        pred = pred.reshape(true.shape)
        u = pred[..., 0]
        v = pred[..., 1]
        A = pred[..., 2]
        
        # Data
        if self.use_data_loss:
            loss_data = self.data_loss(pred, true)
        else:
            loss_data = 0
        # IC
        if self.use_ic_loss:
            loss_ic = self.ic_loss(pred, inputs)
        else:
            loss_ic = 0
            
        # PDE
        if self.use_pde_loss:
            Du, Dv, DA = self.mhd_pde(u, v, A)
            loss_pde = self.mhd_pde_loss(Du, Dv, DA)
        else:
            loss_pde = 0
        
        # Constraints
        if self.use_constraint_loss:
            div_vel, div_B = self.mhd_constraint(self, u, v, A)
            loss_constraints = self.mhd_constraint_loss(div_vel, div_B)
        else:
            loss_constraints = 0

        loss = self.data_weight*loss_data + self.ic_weight*loss_ic + self.pde_weight*loss_pde + self.constraint_weight*loss_constraints
        return loss
    
    def compute_losses(self, pred, true, inputs):
        pred = pred.reshape(true.shape)
        u = pred[..., 0]
        v = pred[..., 1]
        A = pred[..., 2]
        
        loss_dict = {}
        
        # Data
        if self.use_data_loss:
            loss_data, loss_u, loss_v, loss_A = self.data_loss(pred, true, return_all_losses=True)
            loss_dict['loss_data'] = loss_data
            loss_dict['loss_u'] = loss_u
            loss_dict['loss_v'] = loss_v
            loss_dict['loss_A'] = loss_A
        else:
            loss_data = 0
        # IC
        if self.use_ic_loss:
            loss_ic, loss_u_ic, loss_v_ic, loss_A_ic = self.ic_loss(pred, inputs, return_all_losses=True)
            loss_dict['loss_ic'] = loss_ic
            loss_dict['loss_u_ic'] = loss_u_ic
            loss_dict['loss_v_ic'] = loss_v_ic
            loss_dict['loss_A_ic'] = loss_A_ic
        else:
            loss_ic = 0
            
        # PDE
        if self.use_pde_loss:
            Du, Dv, DA = self.mhd_pde(u, v, A)
            loss_pde, loss_Du, loss_Dv, loss_DA = self.mhd_pde_loss(Du, Dv, DA, return_all_losses=True)
            loss_dict['loss_pde'] = loss_pde
            loss_dict['loss_Du'] = loss_Du
            loss_dict['loss_Dv'] = loss_Dv
            loss_dict['loss_DA'] = loss_DA
        else:
            loss_pde = 0
        
        # Constraints
        if self.use_constraint_loss:
            div_vel, div_B = self.mhd_constraint(self, u, v, A)
            loss_constraint, loss_div_vel, loss_div_B = self.mhd_constraint_loss(div_vel, div_B, return_all_losses=True)
            loss_dict['loss_constraint'] = loss_constraint
            loss_dict['loss_div_vel'] = loss_div_vel
            loss_dict['loss_div_B'] = loss_div_B
        else:
            loss_constraints = 0

        loss = self.data_weight*loss_data + self.ic_weight*loss_ic + self.pde_weight*loss_pde + self.constraint_weight*loss_constraints
        loss_dict['loss'] = loss
        return loss, loss_dict
    
    def data_loss(self, pred, true, return_all_losses=False):
        lploss = LpLoss(size_average=True)
        u_pred = pred[..., 0]
        v_pred = pred[..., 1]
        A_pred = pred[..., 2]
        
        u_true = true[..., 0]
        v_true = true[..., 1]
        A_true = true[..., 2]
        
        loss_u = lploss(u_pred, u_true)
        loss_v = lploss(v_pred, v_true)
        loss_A = lploss(A_pred, A_true)
        
        loss_data = self.u_weight*loss_u + self.v_weight*loss_v + self.A_weight*loss_A
        if return_all_losses:
            return loss_data, loss_u, loss_v, loss_A
        else:
            return loss_data
    
    def ic_loss(self, pred, input, return_all_losses=False):
        lploss = LpLoss(size_average=True)
        ic_pred = pred[:, 0]
        ic_true = input[:, 0, ..., 3:]
        u_ic_pred = ic_pred[..., 0]
        v_ic_pred = ic_pred[..., 1]
        Bx_ic_pred = ic_pred[..., 2]
        By_ic_pred = ic_pred[..., 3]
        
        u_ic_true = ic_true[..., 0]
        v_ic_true = ic_true[..., 1]
        Bx_ic_true = ic_true[..., 2]
        By_ic_true = ic_true[..., 3]
        
        loss_u_ic = lploss(u_ic_pred, u_ic_true)
        loss_v_ic = lploss(v_ic_pred, v_ic_true)
        loss_Bx_ic = lploss(Bx_ic_pred, Bx_ic_true)
        loss_By_ic = lploss(By_ic_pred, By_ic_true)
        
        loss_ic = self.u_weight*loss_u_ic + self.v_weight*loss_v_ic + self.Bx_weight*loss_Bx_ic + self.By_weight*loss_By_ic
        if return_all_losses:
            return loss_ic, loss_u_ic, loss_v_ic, loss_Bx_ic, loss_By_ic
        else:
            return loss_ic
    
    def mhd_pde_loss(self, Du, Dv, DA, return_all_losses=None):
        Du_val = torch.zeros_like(Du)
        Dv_val = torch.zeros_like(Dv)
        DA_val = torch.zeros_like(DA)

        loss_Du = F.mse_loss(Du, Du_val)
        loss_Dv = F.mse_loss(Dv, Dv_val)
        loss_DA = F.mse_loss(DA, DA_val)
        
        loss_pde = self.Du_weight*loss_Du + self.Dv_weight*loss_Dv + self.DA_weight*loss_DA
        if return_all_losses:
            return loss_pde, loss_Du, loss_Dv, loss_DA
        else:
            return loss_pde
    
    def mhd_constraint(self, u, v, A):
        batchsize = u.size(0)
        nt = u.size(1)
        nx = u.size(2)
        ny = u.size(3)
        device = u.device
        dt = self.tend / nt
        dx = self.Lx / nx
        dy = self.Ly / ny
        k_max = nx//2
        k_x = 2*np.pi/self.Lx * torch.cat([torch.arange(start=0, end=k_max, step=1, device=device),
                                           torch.arange(start=-k_max, end=0, step=1, device=device)], 0).reshape(nx, 1).repeat(1, ny).reshape(1,1,nx,ny)
        k_y = 2*np.pi/self.Ly * torch.cat([torch.arange(start=0, end=k_max, step=1, device=device),
                                           torch.arange(start=-k_max, end=0, step=1, device=device)], 0).reshape(1, ny).repeat(nx, 1).reshape(1,1,nx,ny)
        
        u_h = torch.fft.fftn(u, dim=[2, 3])
        v_h = torch.fft.fftn(v, dim=[2, 3])
        A_h = torch.fft.fftn(A, dim=[2, 3])
        
        ux_h = self.Du_i(u_h, k_x)
        vy_h = self.Du_i(v_h, k_y)
        
        
        Ax_h = self.Du_i(A_h, k_x)
        Ay_h = self.Du_i(A_h, k_y)
        
        Bx_h = Ay_h
        By_h = -Ax_h      
        
        Bx_x_h = self.Du_i(Bx_h, k_x)
        By_y_h = self.Du_i(By_h, k_y)
          
        ux = torch.irfftn(ux_h[..., :k_max+1], dim=[2, 3])
        vy = torch.irfftn(vy_h[..., :k_max+1], dim=[2, 3])
        Bx_x = torch.irfftn(Bx_x_h[..., :k_max+1], dim=[2, 3])
        By_y = torch.irfftn(By_y_h[..., :k_max+1], dim=[2, 3])
        
        div_vel = ux + vy
        div_B = Bx_x + By_y
        
        return div_vel, div_B
        
    def mhd_pde(self, u, v, A):
        batchsize = u.size(0)
        nx = u.size(1)
        ny = u.size(2)
        nt = u.size(3)
        device = u.device
        dt = self.tend / nt
        dx = self.Lx / nx
        k_max = nx//2
        k_x = 2*np.pi/self.Lx * torch.cat([torch.arange(start=0, end=k_max, step=1, device=device),
                                           torch.arange(start=-k_max, end=0, step=1, device=device)], 0).reshape(nx, 1).repeat(1, ny).reshape(1,1,nx,ny)
        k_y = 2*np.pi/self.Ly * torch.cat([torch.arange(start=0, end=k_max, step=1, device=device),
                                           torch.arange(start=-k_max, end=0, step=1, device=device)], 0).reshape(1, ny).repeat(nx, 1).reshape(1,1,nx,ny)
        lap = -(k_x**2 + k_y**2)
        lap[0, 0, 0, 0] = -1.0
        
        
        u_h = torch.fft.fftn(u, dim=[2, 3])
        v_h = torch.fft.fftn(v, dim=[2, 3])
        A_h = torch.fft.fftn(A, dim=[2, 3])
        
        
        ux_h = self.Du_i(u_h, k_x)
        uy_h = self.Du_i(u_h, k_y)
        vx_h = self.Du_i(v_h, k_x)
        vy_h = self.Du_i(v_h, k_y)
        
        
        Ax_h = self.Du_i(A_h, k_x)
        Ay_h = self.Du_i(A_h, k_y)
        
        Bx_h = Ay_h
        By_h = -Ax_h
        
        B2_h = Bx_h**2 + By_h**2
        
        Bx_x_h = self.Du_i(Bx_h, k_x)
        Bx_y_h = self.Du_i(Bx_h, k_y)
        By_x_h = self.Du_i(By_h, k_x)
        By_y_h = self.Du_i(By_h, k_y)
        
        u_lap_h = self.Lap(u_h, k_x, k_y)
        v_lap_h = self.Lap(v_h, k_x, k_y)
        A_lap_h = self.Lap(A_h, k_x, k_y)
        
        p_h = -self.rho0*(ux_h**2 + 2*uy_h*vx_h + vy_h**2)/lap
        ptot_h = p_h + B2_h/2.0
        ptot_x_h = self.Du_i(ptot_h, k_x)
        ptot_y_h = self.Du_i(ptot_h, k_y)
        
        ux = torch.irfftn(ux_h[..., :k_max+1], dim=[2, 3])
        uy = torch.irfftn(uy_h[..., :k_max+1], dim=[2, 3])
        vx = torch.irfftn(vx_h[..., :k_max+1], dim=[2, 3])
        vy = torch.irfftn(vy_h[..., :k_max+1], dim=[2, 3])
        Ax = torch.irfftn(Ax_h[..., :k_max+1], dim=[2, 3])
        Ay = torch.irfftn(Ay_h[..., :k_max+1], dim=[2, 3])
        Bx = torch.irfftn(Bx_h[..., :k_max+1], dim=[2, 3])
        By = torch.irfftn(By_h[..., :k_max+1], dim=[2, 3])
        B2 = torch.irfftn(B2_h[..., :k_max+1], dim=[2, 3])
        Bx_x = torch.irfftn(Bx_x_h[..., :k_max+1], dim=[2, 3])
        Bx_y = torch.irfftn(Bx_y_h[..., :k_max+1], dim=[2, 3])
        By_x = torch.irfftn(By_x_h[..., :k_max+1], dim=[2, 3])
        By_y = torch.irfftn(By_y_h[..., :k_max+1], dim=[2, 3])
        u_lap = torch.irfftn(u_lap_h[..., :k_max+1], dim=[2, 3])
        v_lap = torch.irfftn(v_lap_h[..., :k_max+1], dim=[2, 3])
        A_lap = torch.irfftn(A_lap_h[..., :k_max+1], dim=[2, 3])
        p = torch.irfftn(p_h[..., :k_max+1], dim=[2, 3])
        ptot = torch.irfftn(ptot_h[..., :k_max+1], dim=[2, 3])
        ptot_x = torch.irfftn(ptot_x_h[..., :k_max+1], dim=[2, 3])
        ptot_y = torch.irfftn(ptot_y_h[..., :k_max+1], dim=[2, 3])
        
        vel_grad_u = u*ux + v*uy
        vel_grad_v = u*vx + v*vy
        
        B_grad_Bx = Bx*Bx_x + By*Bx_y 
        B_grad_By = Bx*By_x + By*By_y
        
        vel_grad_A = u*Ax + v*Ay
        
        u_rhs = -vel_grad_u - ptot_x/self.rho0 + B_grad_Bx/self.rho0 + self.nu*u_lap
        v_rhs = -vel_grad_v - ptot_y/self.rho0 + B_grad_By/self.rho0 + self.nu*v_lap
        A_rhs = -vel_grad_A + self.eta*A_lap
        
        u_t = self.Du_t(u, dt)
        v_t = self.Du_t(v, dt)
        A_t = self.Du_t(A, dt)
        
        Du = u_t - u_rhs[:, 1:-1]
        Dv = v_t - v_rhs[:, 1:-1]
        DA = A_t - A_rhs[:, 1:-1]
        
        return Du, Dv, DA
        

        
def FDM_swe_nonlin(h, u, v, D=1, g=1.0, nu=1.0e-3):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)
    h = h.reshape(batchsize, nx, ny, nt)
    u = u.reshape(batchsize, nx, ny, nt)
    v = v.reshape(batchsize, nx, ny, nt)
    dt = D / (nt-1)
    dx = D / (nx)
#     u2 = u**2
    hu = h*u
    hv = h*v
    huu = h*u**2
    huv = h*u*v
    hvv = h*v**2
    hh = h**2
    hu_h = torch.fft.fftn(hu, dim=[1, 2])
    hv_h = torch.fft.fftn(hv, dim=[1, 2])
    huu_h = torch.fft.fftn(huu, dim=[1, 2])
    huv_h = torch.fft.fftn(huv, dim=[1, 2])
    hvv_h = torch.fft.fftn(hvv, dim=[1, 2])
    hh_h = torch.fft.fftn(hh, dim=[1, 2])
    u_h = torch.fft.fftn(u, dim=[1, 2])
    v_h = torch.fft.fftn(v, dim=[1, 2])
#     u2_h = torch.fft.fftn(u2, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=h.device),
                     torch.arange(start=-k_max, end=0, step=1, device=h.device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=h.device),
                     torch.arange(start=-k_max, end=0, step=1, device=h.device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
    
    hux_h = 2j*np.pi*k_x*hu_h
    hvy_h = 2j*np.pi*k_y*hv_h
    
    huux_h = 2j*np.pi*k_x*huu_h
    hvvy_h = 2j*np.pi*k_y*hvv_h

    huvx_h = 2j*np.pi*k_x*huv_h
    huvy_h = 2j*np.pi*k_y*huv_h
    
    hhx_h = 2j*np.pi*k_x*hh_h
    hhy_h = 2j*np.pi*k_y*hh_h
    
    
    ux_h = 2j*np.pi*k_x*u_h
    uxx_h = 2j*np.pi*k_x*ux_h
    uy_h = 2j*np.pi*k_y*u_h
    uyy_h = 2j*np.pi*k_y*uy_h
    
    vx_h = 2j*np.pi*k_x*v_h
    vxx_h = 2j*np.pi*k_x*vx_h
    vy_h = 2j*np.pi*k_y*v_h
    vyy_h =2j *np.pi*k_y*vy_h


#     hx = torch.fft.irfftn(hx_h[:, :, :k_max+1], dim=[1, 2])
#     hy = torch.fft.irfftn(hy_h[:, :, :k_max+1], dim=[1, 2])
    hux = torch.fft.irfftn(hux_h[:, :, :k_max+1], dim=[1, 2])
    hvy = torch.fft.irfftn(hvy_h[:, :, :k_max+1], dim=[1, 2])
    huux = torch.fft.irfftn(huux_h[:, :, :k_max+1], dim=[1, 2])
    hvvy = torch.fft.irfftn(hvvy_h[:, :, :k_max+1], dim=[1, 2])
    huvx = torch.fft.irfftn(huvx_h[:, :, :k_max+1], dim=[1, 2])
    huvy = torch.fft.irfftn(huvy_h[:, :, :k_max+1], dim=[1, 2])
    hhx = torch.fft.irfftn(hhx_h[:, :, :k_max+1], dim=[1, 2])
    hhy = torch.fft.irfftn(hhy_h[:, :, :k_max+1], dim=[1, 2])
    ht = (h[..., 2:] - h[..., :-2]) / (2 * dt)
    
    ux = torch.fft.irfftn(ux_h[:, :, :k_max+1], dim=[1, 2])
    uy = torch.fft.irfftn(uy_h[:, :, :k_max+1], dim=[1, 2])
    uxx = torch.fft.irfftn(uxx_h[:, :, :k_max+1], dim=[1, 2])
    uyy = torch.fft.irfftn(uyy_h[:, :, :k_max+1], dim=[1, 2])
    ut = (u[..., 2:] - u[..., :-2]) / (2 * dt)
    
    vx = torch.fft.irfftn(vx_h[:, :, :k_max+1], dim=[1, 2])
    vy = torch.fft.irfftn(vy_h[:, :, :k_max+1], dim=[1, 2])
    vxx = torch.fft.irfftn(vxx_h[:, :, :k_max+1], dim=[1, 2])
    vyy = torch.fft.irfftn(vyy_h[:, :, :k_max+1], dim=[1, 2])
    vt = (v[..., 2:] - v[..., :-2]) / (2 * dt)
#     utt = (u[..., 2:] - 2.0*u[..., 1:-1] + u[..., :-2]) / (dt**2)
#     Du = ut + (0.5*(u2x + u2y) - nu*(uxx + uyy))[..., 1:-1]
    Dh = ht + (hux + hvy)[..., 1:-1]
    Du = ut + ((huux + 0.5*g*hhx) + huvy - nu*(uxx + uyy))[..., 1:-1]
    Dv = vt + (huvx + (hvvy + 0.5*g*hhy) - nu*(vxx + vyy))[..., 1:-1]
#     Du = ut + (u*(ux + uy) - nu*(uxx + uyy))[..., 1:-1]
    return Dh, Du, Dv

def swe_loss(s_pred, s_true, H=1.0, use_sum=True):
    h_pred = s_pred[..., 0] - H
    u_pred = s_pred[..., 1]
    v_pred = s_pred[..., 2]
    h_true = s_true[..., 0] - H
    u_true = s_true[..., 1]
    v_true = s_true[..., 2]
    lploss = LpLoss(size_average=True)
    loss_h = lploss(h_pred, h_true)
    loss_u = lploss(u_pred, u_true)
    loss_v = lploss(v_pred, v_true)
    loss_s = torch.stack([loss_h, loss_u, loss_v], dim=-1)
    if use_sum:
        data_loss = torch.sum(loss_s)
    else:
        data_loss = torch.mean(loss_s)
    return data_loss
    
    

def PINO_loss_swe_nonlin(s, s0, g=1.0, nu=0.001, h_weight=1.0, u_weight=1.0, v_weight=1.0):
    batchsize = s.size(0)
    nx = s.size(1)
    ny = s.size(2)
    nt = s.size(3)
    s = s.reshape(batchsize, nx, ny, nt, 3)

    lploss = LpLoss(size_average=True)
    s_ic = s[..., 0, 0].reshape(s0.shape)
    loss_ic = lploss(s_ic, s0)
#     index_t = torch.zeros(nx,).long()
#     index_x = torch.tensor(range(nx)).long()
#     boundary_u = u[:, index_t, index_x]
#     loss_u = F.mse_loss(boundary_u, u0)

#     Du = FDM_Burgers(u, nu=nu)[:, :, :, :]
    h = s[..., 0]
    u = s[..., 1]
    v = s[..., 2]
    Dh, Du, Dv = FDM_swe_nonlin(h, u, v, g=g, nu=nu)
    Dh *= h_weight
    Du *= u_weight
    Dv *= v_weight
    Ds = torch.stack([Dh, Du, Dv], dim=-1)
    f_ = torch.zeros(Ds.shape, device=s.device) # use f_ to distinguish from corriolous const f
    loss_f = F.mse_loss(Ds, f_)

    # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
    # loss_bc1 = F.mse_loss((u[:, :, 1] - u[:, :, -1]) /
    #                       (2/(nx)), (u[:, :, 0] - u[:, :, -2])/(2/(nx)))
    return loss_ic, loss_f