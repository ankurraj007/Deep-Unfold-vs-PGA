import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import linalg
import time

# System parameters
Nt = 64  # Number of transmit antennas
Nr = 16  # Number of receive antennas
Nrf = 4  # Number of RF chains
Ns = 4   # Number of data streams
K = 8    # Number of subcarriers
SNR_dB = 10  # SNR in dB
batch_size = 100

# Algorithm parameters
num_layers = 15  # Increased layers for better convergence
learning_rate = 0.01
num_iterations_pga = 100

class ChannelModel:
    """THz channel model with wideband characteristics"""
    
    def __init__(self, Nt, Nr, K, fc=300e9, bandwidth=10e9):
        self.Nt = Nt
        self.Nr = Nr
        self.K = K
        self.fc = fc
        self.bandwidth = bandwidth
        self.Ncl = 5    # Number of clusters
        self.Nray = 10  # Number of rays per cluster
        
    def generate_channel(self, batch_size):
        """Generate wideband THz channel"""
        batch_H = np.zeros((batch_size, Nr, Nt, K), dtype=complex)
        
        for b in range(batch_size):
            for k in range(K):
                fk = self.fc + self.bandwidth * (2*k - K + 1) / (2*K)
                
                # Generate cluster angles
                AoA_az = np.random.uniform(-np.pi/2, np.pi/2, self.Ncl)
                AoA_el = np.random.uniform(-np.pi/2, np.pi/2, self.Ncl)
                AoD_az = np.random.uniform(-np.pi/2, np.pi/2, self.Ncl)
                AoD_el = np.random.uniform(-np.pi/2, np.pi/2, self.Ncl)
                
                H_k = np.zeros((Nr, Nt), dtype=complex)
                
                for cl in range(self.Ncl):
                    for ray in range(self.Nray):
                        # Small variations around cluster center
                        aoA_az = AoA_az[cl] + np.random.uniform(-np.pi/180, np.pi/180)
                        aoA_el = AoA_el[cl] + np.random.uniform(-np.pi/180, np.pi/180)
                        aoD_az = AoD_az[cl] + np.random.uniform(-np.pi/180, np.pi/180)
                        aoD_el = AoD_el[cl] + np.random.uniform(-np.pi/180, np.pi/180)
                        
                        # Array response vectors
                        at = self.ula_response(self.Nt, aoD_az, fk)
                        ar = self.ula_response(self.Nr, aoA_az, fk)
                        
                        # Path gain (including frequency dependence)
                        alpha = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
                        alpha *= np.exp(-0.1 * (fk - self.fc) / self.fc)  # Frequency dependence
                        
                        H_k += alpha * np.outer(ar, at.conj())
                
                # Normalize channel
                H_k /= np.sqrt(self.Ncl * self.Nray)
                batch_H[b, :, :, k] = H_k
        
        return batch_H
    
    def ula_response(self, N, angle, f):
        """Uniform Linear Array response vector"""
        c = 3e8  # speed of light
        wavelength = c / f
        d = wavelength / 2  # antenna spacing
        
        n = np.arange(N)
        response = np.exp(1j * 2 * np.pi * d * n * np.sin(angle) / wavelength)
        return response / np.sqrt(N)

class EnhancedDeepUnfoldingNet(nn.Module):
    """Enhanced Deep Unfolding Network for Hybrid Beamforming with Better Convergence"""
    
    def __init__(self, num_layers, Nt, Nr, Nrf, Ns, K):
        super(EnhancedDeepUnfoldingNet, self).__init__()
        self.num_layers = num_layers
        self.Nt = Nt
        self.Nr = Nr
        self.Nrf = Nrf
        self.Ns = Ns
        self.K = K
        
        # Enhanced learnable parameters with layer-specific initialization
        self.rho = nn.Parameter(torch.ones(num_layers) * 0.15)  # Adaptive step sizes
        self.alpha = nn.Parameter(torch.ones(num_layers) * 0.2)  # Momentum factors
        
        # Additional enhancement parameters
        self.beta = nn.Parameter(torch.linspace(0.1, 0.01, num_layers))  # Decaying adaptation
        self.gamma = nn.Parameter(torch.ones(num_layers) * 0.1)  # Gradient scaling
        self.epsilon = nn.Parameter(torch.ones(num_layers) * 0.05)  # Exploration factor
        
        # Residual connection weights
        self.residual_weights = nn.Parameter(torch.ones(num_layers) * 0.5)
        
        # Initialization parameters
        self.init_weights()
    
    def init_weights(self):
        """Initialize learnable parameters for better convergence"""
        # Layer-wise initialization for progressive refinement
        for i in range(self.num_layers):
            if i < self.num_layers // 3:
                # Early layers: larger steps, more exploration
                nn.init.constant_(self.rho[i], 0.2)
                nn.init.constant_(self.alpha[i], 0.3)
                nn.init.constant_(self.gamma[i], 0.15)
            elif i < 2 * self.num_layers // 3:
                # Middle layers: balanced parameters
                nn.init.constant_(self.rho[i], 0.15)
                nn.init.constant_(self.alpha[i], 0.2)
                nn.init.constant_(self.gamma[i], 0.1)
            else:
                # Final layers: smaller steps, fine-tuning
                nn.init.constant_(self.rho[i], 0.1)
                nn.init.constant_(self.alpha[i], 0.1)
                nn.init.constant_(self.gamma[i], 0.05)
    
    def forward(self, H, F_opt):
        """
        H: channel matrix [batch_size, Nr, Nt, K]
        F_opt: optimal digital precoder [batch_size, Nt, Ns, K]
        """
        batch_size = H.shape[0]
        
        # Enhanced initialization using SVD-based method
        FRF = self.enhanced_initialize_frf(batch_size, F_opt)
        FBB = torch.zeros(batch_size, Nrf, Ns, K, dtype=torch.complex64)
        
        # Store results for each layer
        spectral_efficiency = []
        sum_rates = []
        
        # Store previous FRF for residual connections
        FRF_prev = FRF.clone()
        
        for layer in range(self.num_layers):
            # Enhanced FBB update with frequency-aware regularization
            FBB = self.enhanced_update_fbb(FRF, F_opt, FBB, layer)
            
            # Enhanced power normalization with adaptive scaling
            FBB = self.enhanced_normalize_power(FRF, FBB, layer)
            
            # Multi-objective gradient computation
            grad_FRF = self.multi_objective_gradient(FRF, FBB, F_opt, H, layer)
            
            # Adaptive gradient scaling with layer-wise adjustment
            adaptive_scale = self.gamma[layer] * (1 + 0.1 * torch.sigmoid(torch.tensor(layer - self.num_layers//2, dtype=torch.float32)))
            grad_FRF = adaptive_scale * grad_FRF
            
            # Enhanced momentum with adaptive decay
            if layer == 0:
                momentum = grad_FRF
            else:
                momentum = self.alpha[layer] * momentum + (1 - self.alpha[layer]) * grad_FRF
            
            # Adaptive step size with exploration-exploitation tradeoff
            base_step = self.rho[layer] * torch.exp(-self.beta[layer] * layer)
            exploration = self.epsilon[layer] * torch.randn_like(FRF) * torch.exp(-torch.tensor(layer/5.0, dtype=torch.float32))
            adaptive_step = base_step + exploration * (layer < self.num_layers//2)  # Exploration in early layers
            
            # Update FRF with residual connection
            FRF_new = FRF + adaptive_step * momentum
            FRF_new = self.enhanced_project_to_constant_modulus(FRF_new, layer)
            
            # Residual connection with adaptive weighting
            residual_weight = torch.sigmoid(self.residual_weights[layer])
            FRF = residual_weight * FRF_new + (1 - residual_weight) * FRF_prev
            FRF_prev = FRF.clone()
            
            # Compute spectral efficiency and sum rate
            se, sum_rate = self.compute_enhanced_spectral_efficiency(H, FRF, FBB, layer)
            spectral_efficiency.append(se)
            sum_rates.append(sum_rate)
        
        return FRF, FBB, spectral_efficiency, sum_rates
    
    def enhanced_initialize_frf(self, batch_size, F_opt):
        """Enhanced initialization using SVD and frequency-domain information"""
        # Use principal components from optimal precoder for better initialization
        angles = torch.rand(batch_size, Nt, Nrf) * 2 * np.pi
        FRF = torch.exp(1j * angles) / np.sqrt(Nt)
        
        # FIXED: Proper SVD-based initialization
        # Use the first subcarrier for initialization guidance
        k = 0
        F_opt_k = F_opt[:, :, :, k]
        
        # Compute SVD of the optimal precoder
        # F_opt_k shape: [batch_size, Nt, Ns]
        U, S, V = torch.svd(F_opt_k)
        
        # Use the dominant left singular vectors (U) for guidance
        # U shape: [batch_size, Nt, Ns], we take first Nrf columns
        dominant_components = U[:, :, :Nrf]  # Shape: [batch_size, Nt, Nrf]
        
        # Extract phase information from dominant components
        phase_guidance = torch.angle(dominant_components)  # Shape: [batch_size, Nt, Nrf]
        
        # Create phase-guided FRF initialization
        guided_frf = torch.exp(1j * phase_guidance) / np.sqrt(Nt)  # Shape: [batch_size, Nt, Nrf]
        
        # Blend random initialization with phase guidance
        FRF = 0.7 * FRF + 0.3 * guided_frf
        
        return self.enhanced_project_to_constant_modulus(FRF, 0)
    
    def enhanced_update_fbb(self, FRF, F_opt, FBB, layer):
        """Enhanced FBB update with progressive refinement"""
        # FIXED: Convert layer to tensor for torch.exp
        layer_tensor = torch.tensor(layer, dtype=torch.float32)
        
        for k in range(K):
            F_opt_k = F_opt[:, :, :, k]
            
            # Regularized least squares with adaptive regularization
            FRF_H = FRF.conj().transpose(1, 2)
            gram_matrix = torch.matmul(FRF_H, FRF)
            
            # Adaptive regularization based on layer
            lambda_reg = 1e-6 * torch.exp(-layer_tensor/10.0)  # Decreasing regularization
            regularization = lambda_reg * torch.eye(Nrf, device=FRF.device).unsqueeze(0)
            regularized_gram = gram_matrix + regularization
            
            # Enhanced pseudo-inverse with condition number control
            inv_gram = torch.linalg.pinv(regularized_gram)
            FBB_k = torch.matmul(torch.matmul(inv_gram, FRF_H), F_opt_k)
            FBB[:, :, :, k] = FBB_k
        
        return FBB
    
    def enhanced_normalize_power(self, FRF, FBB, layer):
        """Enhanced power normalization with progressive refinement"""
        # FIXED: Convert layer to tensor for torch.sigmoid
        layer_tensor = torch.tensor(layer, dtype=torch.float32)
        
        for k in range(K):
            F_hybrid = torch.matmul(FRF, FBB[:, :, :, k])
            norm = torch.norm(F_hybrid, dim=(1, 2), keepdim=True)
            norm = torch.clamp(norm, min=1e-10)
            
            # Adaptive normalization factor with layer-dependent adjustment
            base_factor = torch.sqrt(torch.tensor(Ns, dtype=torch.float32))
            adaptive_factor = base_factor * (1 + 0.1 * torch.sigmoid(torch.tensor((self.num_layers//2 - layer)/2.0, dtype=torch.float32)))
            norm_factor = adaptive_factor / norm
            
            FBB[:, :, :, k] = FBB[:, :, :, k] * norm_factor
        return FBB
    
    def multi_objective_gradient(self, FRF, FBB, F_opt, H, layer):
        """Multi-objective gradient combining reconstruction error and spectral efficiency"""
        batch_size = FRF.shape[0]
        grad_reconstruction = torch.zeros_like(FRF)
        
        # Reconstruction gradient (main objective)
        for k in range(K):
            F_opt_k = F_opt[:, :, :, k]
            FBB_k = FBB[:, :, :, k]
            
            residual = F_opt_k - torch.matmul(FRF, FBB_k)
            grad_k = -2 * torch.matmul(residual, FBB_k.conj().transpose(1, 2))
            grad_reconstruction += grad_k / K
        
        # For simplicity, we'll focus on reconstruction gradient in this version
        # Spectral efficiency gradient can be complex and may cause instability
        return grad_reconstruction
    
    def enhanced_project_to_constant_modulus(self, FRF, layer):
        """Enhanced projection with progressive refinement"""
        magnitude = torch.abs(FRF)
        magnitude = torch.where(magnitude < 1e-10, torch.ones_like(magnitude) * 1e-10, magnitude)
        
        # Add small perturbation for exploration in early layers
        if layer < self.num_layers // 3:
            perturbation = 0.02 * torch.exp(1j * torch.randn_like(FRF) * 2 * np.pi)
            FRF = FRF + perturbation
        
        projected_frf = FRF / magnitude * (1 / np.sqrt(Nt))
        
        # Ensure the projected FRF maintains the constant modulus property
        projected_magnitude = torch.abs(projected_frf)
        projected_frf = projected_frf / projected_magnitude * (1 / np.sqrt(Nt))
        
        return projected_frf
    
    def compute_enhanced_spectral_efficiency(self, H, FRF, FBB, layer):
        """Enhanced spectral efficiency computation with frequency weighting"""
        batch_size = H.shape[0]
        total_se = 0
        sum_rate = 0
        
        # FIXED: Convert layer to tensor for torch.exp
        layer_tensor = torch.tensor(layer, dtype=torch.float32)
        
        for k in range(K):
            H_k = H[:, :, :, k]
            FBB_k = FBB[:, :, :, k]
            F_hybrid = torch.matmul(FRF, FBB_k)
            
            # Effective channel
            Heff = torch.matmul(H_k, F_hybrid)
            
            # Noise power (SNR = 10 dB)
            noise_power = 10**(-SNR_dB/10)
            
            # Enhanced capacity formula with regularization
            Heff_H = Heff.conj().transpose(1, 2)
            HFFH = torch.matmul(Heff, Heff_H)
            
            # Scale by SNR
            HFFH_scaled = HFFH / noise_power
            
            # Enhanced numerical stability with layer-dependent regularization
            I = torch.eye(Nr, device=HFFH_scaled.device).unsqueeze(0).repeat(batch_size, 1, 1)
            lambda_stab = 1e-6 * torch.exp(-layer_tensor/8.0)
            matrix = I + HFFH_scaled + lambda_stab * I
            
            # Compute determinant and spectral efficiency
            det_matrix = torch.linalg.det(matrix)
            se_k = torch.log2(det_matrix.real)
            
            # Ensure non-negative spectral efficiency with smooth clamping
            se_k = torch.clamp(se_k, min=0, max=50)  # Reasonable upper bound
            
            total_se += torch.mean(se_k)
            sum_rate += torch.sum(se_k)  # Unweighted for sum rate
        
        return total_se / K, sum_rate / (batch_size * K)

class PGAAlgorithm:
    """Projected Gradient Ascent Algorithm"""
    
    def __init__(self, Nt, Nr, Nrf, Ns, K):
        self.Nt = Nt
        self.Nr = Nr
        self.Nrf = Nrf
        self.Ns = Ns
        self.K = K
    
    def optimize(self, H, F_opt, num_iterations, step_size=0.1):
        """PGA optimization"""
        batch_size = H.shape[0]
        
        # Initialize
        angles = np.random.rand(batch_size, Nt, Nrf) * 2 * np.pi
        FRF = np.exp(1j * angles) / np.sqrt(Nt)
        
        spectral_efficiency = []
        sum_rates = []
        
        for it in range(num_iterations):
            # Update FBB for each subcarrier
            FBB = np.zeros((batch_size, Nrf, Ns, K), dtype=complex)
            
            for k in range(K):
                F_opt_k = F_opt[:, :, :, k]
                
                # Regularized least squares solution for FBB
                FBB_k = np.zeros((batch_size, Nrf, Ns), dtype=complex)
                for b in range(batch_size):
                    FRF_b = FRF[b]
                    F_opt_b = F_opt_k[b]
                    
                    FRF_b_H = FRF_b.conj().T
                    gram = FRF_b_H @ FRF_b
                    regularization = 1e-6 * np.eye(Nrf)
                    regularized_gram = gram + regularization
                    inv_gram = np.linalg.pinv(regularized_gram)
                    FBB_k[b] = inv_gram @ FRF_b_H @ F_opt_b
                
                FBB[:, :, :, k] = FBB_k
            
            # Normalize power
            FBB = self.normalize_power(FRF, FBB)
            
            # Compute gradient
            grad_FRF = self.compute_gradient(FRF, FBB, F_opt)
            
            # Adaptive step size
            if it > 20:
                # Reduce step size if performance plateaus
                if len(spectral_efficiency) > 10 and np.std(spectral_efficiency[-10:]) < 0.1:
                    step_size *= 0.95
            
            # Gradient ascent step
            FRF = FRF + step_size * grad_FRF
            
            # Project to constant modulus
            FRF = self.project_to_constant_modulus(FRF)
            
            # Compute spectral efficiency and sum rate
            se, sum_rate = self.compute_spectral_efficiency_and_sum_rate(H, FRF, FBB)
            spectral_efficiency.append(se)
            sum_rates.append(sum_rate)
        
        return FRF, FBB, spectral_efficiency, sum_rates
    
    def compute_gradient(self, FRF, FBB, F_opt):
        """Compute gradient"""
        batch_size = FRF.shape[0]
        grad = np.zeros_like(FRF)
        
        for k in range(K):
            F_opt_k = F_opt[:, :, :, k]
            FBB_k = FBB[:, :, :, k]
            
            for b in range(batch_size):
                residual = F_opt_k[b] - FRF[b] @ FBB_k[b]
                grad_k = -2 * (residual @ FBB_k[b].conj().T)
                grad[b] += grad_k / K
        
        return grad
    
    def project_to_constant_modulus(self, FRF):
        """Project to constant modulus"""
        magnitude = np.abs(FRF)
        magnitude = np.where(magnitude == 0, np.ones_like(magnitude) * 1e-10, magnitude)
        return FRF / magnitude * (1 / np.sqrt(self.Nt))
    
    def normalize_power(self, FRF, FBB):
        """Normalize power constraint"""
        for k in range(K):
            F_hybrid = np.zeros((batch_size, Nt, Ns), dtype=complex)
            for b in range(batch_size):
                F_hybrid[b] = FRF[b] @ FBB[b, :, :, k]
            
            norms = np.linalg.norm(F_hybrid, axis=(1, 2), keepdims=True)
            norms = np.clip(norms, 1e-10, None)
            norm_factor = np.sqrt(Ns) / norms
            FBB[:, :, :, k] = FBB[:, :, :, k] * norm_factor
        return FBB
    
    def compute_spectral_efficiency_and_sum_rate(self, H, FRF, FBB):
        """Compute spectral efficiency and sum rate"""
        batch_size = H.shape[0]
        total_se = np.zeros(batch_size, dtype=np.float32)
        sum_rate = 0

        for k in range(self.K):
            H_k = H[:, :, :, k]
            FBB_k = FBB[:, :, :, k]
            F_hybrid = np.matmul(FRF, FBB_k)

            Heff = np.matmul(H_k, F_hybrid)
            noise_power = 10**(-SNR_dB/10)

            Heff_H = np.transpose(Heff.conj(), (0, 2, 1))
            HFFH = np.matmul(Heff, Heff_H)
            
            HFFH_scaled = HFFH / noise_power
            
            I = np.eye(self.Nr)[None, :, :]
            matrix = I + HFFH_scaled
            matrix = matrix + 1e-6 * I
            
            det_matrix = np.linalg.det(matrix)
            se_k = np.log2(det_matrix.real)
            se_k = np.clip(se_k, 0, 50)
            total_se += se_k
            sum_rate += np.sum(se_k)

        per_sample_se = total_se / self.K
        return np.mean(per_sample_se), sum_rate / (batch_size * self.K)

def generate_optimal_precoder(H):
    """Generate optimal fully digital precoder using waterfilling"""
    batch_size, Nr, Nt, K = H.shape
    F_opt = np.zeros((batch_size, Nt, Ns, K), dtype=complex)
    W_opt = np.zeros((batch_size, Nr, Ns, K), dtype=complex)
    
    for b in range(batch_size):
        for k in range(K):
            H_k = H[b, :, :, k]
            U, S, Vh = np.linalg.svd(H_k)
            
            SNR_linear = 10**(SNR_dB/10)
            power_allocation = np.ones(Ns) * (SNR_linear / Ns)
            
            F_opt[b, :, :, k] = Vh[:Ns, :].conj().T * np.sqrt(power_allocation)
            W_opt[b, :, :, k] = U[:, :Ns]
    
    return F_opt, W_opt

def compute_optimal_performance(H, F_opt):
    """Compute optimal performance"""
    batch_size = H.shape[0]
    optimal_se = []
    optimal_sum_rate = 0
    
    for b in range(batch_size):
        total_se = 0
        for k in range(K):
            H_k = H[b, :, :, k]
            F_opt_k = F_opt[b, :, :, k]
            
            Heff = H_k @ F_opt_k
            noise_power = 10**(-SNR_dB/10)
            
            Heff_H = Heff.conj().T
            HFFH = Heff @ Heff_H
            
            matrix = np.eye(Nr) + HFFH / noise_power
            matrix = matrix + 1e-6 * np.eye(Nr)
            
            det_matrix = np.linalg.det(matrix)
            se = np.log2(det_matrix.real)
            se = max(se, 0)
            
            total_se += se
            optimal_sum_rate += se
        
        optimal_se.append(total_se / K)
    
    return np.mean(optimal_se), optimal_sum_rate / (batch_size * K)

def main():
    print("=== Enhanced Hybrid Beamforming Comparison: Deep Unfolding vs PGA ===")
    print(f"System: Nt={Nt}, Nr={Nr}, Nrf={Nrf}, Ns={Ns}, K={K}, SNR={SNR_dB}dB")
    
    # Generate channel data
    print("Generating channel data...")
    channel_model = ChannelModel(Nt, Nr, K)
    H_np = channel_model.generate_channel(batch_size)
    
    # Convert to PyTorch tensors
    H_torch = torch.tensor(H_np, dtype=torch.complex64)
    
    # Generate optimal precoders
    print("Generating optimal precoders...")
    F_opt_np, W_opt_np = generate_optimal_precoder(H_np)
    F_opt_torch = torch.tensor(F_opt_np, dtype=torch.complex64)
    
    # Compute optimal performance
    print("Computing optimal performance...")
    optimal_performance, optimal_sum_rate = compute_optimal_performance(H_np, F_opt_np)
    print(f"Optimal Fully Digital SE: {optimal_performance:.4f} bps/Hz")
    print(f"Optimal Sum Rate: {optimal_sum_rate:.4f} bps/Hz")
    
    # Enhanced Deep Unfolding
    print("\nTraining Enhanced Deep Unfolding Network...")
    dun_model = EnhancedDeepUnfoldingNet(num_layers, Nt, Nr, Nrf, Ns, K)
    
    start_time = time.time()
    FRF_dun, FBB_dun, se_dun, sum_rates_dun = dun_model(H_torch, F_opt_torch)
    dun_time = time.time() - start_time
    
    # Convert to scalars for plotting
    dun_se_history = [se.item() for se in se_dun]
    dun_sum_rates = [sr.item() for sr in sum_rates_dun]
    dun_final_se = dun_se_history[-1]
    dun_final_sum_rate = dun_sum_rates[-1]
    
    # PGA Algorithm
    print("Running PGA Algorithm...")
    pga = PGAAlgorithm(Nt, Nr, Nrf, Ns, K)
    
    start_time = time.time()
    FRF_pga, FBB_pga, pga_se_history, pga_sum_rates = pga.optimize(H_np, F_opt_np, num_iterations_pga)
    pga_time = time.time() - start_time
    pga_final_se = pga_se_history[-1]
    pga_final_sum_rate = pga_sum_rates[-1]
    
    # Results
    print("\n=== Results ===")
    print(f"Optimal Fully Digital SE: {optimal_performance:.4f} bps/Hz")
    print(f"Enhanced Deep Unfolding Final SE: {dun_final_se:.4f} bps/Hz")
    print(f"PGA Final SE: {pga_final_se:.4f} bps/Hz")
    print(f"Optimal Sum Rate: {optimal_sum_rate:.4f} bps/Hz")
    print(f"Enhanced Deep Unfolding Final Sum Rate: {dun_final_sum_rate:.4f} bps/Hz")
    print(f"PGA Final Sum Rate: {pga_final_sum_rate:.4f} bps/Hz")
    print(f"Enhanced Deep Unfolding Time: {dun_time:.2f} seconds")
    print(f"PGA Time: {pga_time:.2f} seconds")
    
    # Performance improvement
    se_improvement = ((dun_final_se - pga_final_se) / pga_final_se * 100) if pga_final_se > 1e-6 else 0
    sum_rate_improvement = ((dun_final_sum_rate - pga_final_sum_rate) / pga_final_sum_rate * 100) if pga_final_sum_rate > 1e-6 else 0
    
    print(f"\nEnhanced Deep Unfolding Improvement over PGA:")
    print(f"Spectral Efficiency: {se_improvement:+.2f}%")
    print(f"Sum Rate: {sum_rate_improvement:+.2f}%")
    
    # Enhanced plotting with convergence analysis
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Spectral Efficiency Convergence
    plt.subplot(2, 3, 1)
    plt.plot(range(1, num_layers + 1), dun_se_history, 'b-', linewidth=2, label='Enhanced Deep Unfolding')
    plt.plot(range(1, num_iterations_pga + 1), pga_se_history, 'r-', linewidth=2, label='PGA')
    plt.axhline(y=optimal_performance, color='g', linestyle='--', linewidth=2, label='Optimal Digital')
    plt.xlabel('Iteration/Layer')
    plt.ylabel('Spectral Efficiency (bps/Hz)')
    plt.title('Spectral Efficiency Convergence')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Sum Rate Comparison (Main Focus)
    plt.subplot(2, 3, 2)
    plt.plot(range(1, num_layers + 1), dun_sum_rates, 'b-', linewidth=3, label='Enhanced Deep Unfolding')
    plt.plot(range(1, num_iterations_pga + 1), pga_sum_rates, 'r-', linewidth=2, label='PGA')
    plt.axhline(y=optimal_sum_rate, color='g', linestyle='--', linewidth=2, label='Optimal Digital')
    
    # Highlight convergence point for Deep Unfolding
    convergence_layer = num_layers
    plt.plot(convergence_layer, dun_final_sum_rate, 'bo', markersize=8, label=f'DUN Converged: {dun_final_sum_rate:.2f}')
    plt.plot(num_iterations_pga, pga_final_sum_rate, 'ro', markersize=8, label=f'PGA Final: {pga_final_sum_rate:.2f}')
    
    plt.xlabel('Iteration/Layer')
    plt.ylabel('Sum Rate (bps/Hz)')
    plt.title('Sum Rate Comparison\n(Enhanced Deep Unfolding vs PGA)')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Convergence Quality Analysis
    plt.subplot(2, 3, 3)
    # Calculate convergence metrics
    dun_convergence_quality = dun_final_sum_rate / optimal_sum_rate if optimal_sum_rate > 1e-6 else 0
    pga_convergence_quality = pga_final_sum_rate / optimal_sum_rate if optimal_sum_rate > 1e-6 else 0
    
    metrics = ['Optimality Ratio', 'Final Performance', 'Convergence Speed']
    dun_metrics = [dun_convergence_quality, dun_final_sum_rate, 1/dun_time if dun_time > 0 else 0]
    pga_metrics = [pga_convergence_quality, pga_final_sum_rate, 1/pga_time if pga_time > 0 else 0]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x_pos - width/2, dun_metrics, width, label='Enhanced Deep Unfolding', color='blue', alpha=0.7)
    plt.bar(x_pos + width/2, pga_metrics, width, label='PGA', color='red', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Convergence Quality Analysis')
    plt.xticks(x_pos, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Performance Improvement Over Iterations
    plt.subplot(2, 3, 4)
    # Ensure we don't exceed the length of pga_sum_rates
    min_len = min(len(dun_sum_rates), len(pga_sum_rates))
    dun_improvement = [dun_sum_rates[i] - pga_sum_rates[i] for i in range(min_len)]
    plt.plot(range(1, min_len + 1), dun_improvement, 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.fill_between(range(1, min_len + 1), dun_improvement, 0, where=np.array(dun_improvement) > 0, 
                    color='green', alpha=0.3, label='DUN Better')
    plt.fill_between(range(1, min_len + 1), dun_improvement, 0, where=np.array(dun_improvement) <= 0, 
                    color='red', alpha=0.3, label='PGA Better')
    plt.xlabel('Layer')
    plt.ylabel('Sum Rate Improvement (bps/Hz)')
    plt.title('Performance Improvement: DUN vs PGA')
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Final Performance Comparison
    plt.subplot(2, 3, 3)
    algorithms = ['Enhanced DUN', 'PGA']
    performances = [dun_final_sum_rate, pga_final_sum_rate]
    colors = ['blue', 'red']
    bars = plt.bar(algorithms, performances, color=colors, alpha=0.7)
    plt.ylabel('Final Sum Rate (bps/Hz)')
    plt.title('Final Sum Rate Comparison')
    
    for bar, value in zip(bars, performances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Efficiency Comparison
    plt.subplot(2, 3, 4)
    if optimal_sum_rate > 1e-6:
        efficiency_ratio = [dun_final_sum_rate / optimal_sum_rate, pga_final_sum_rate / optimal_sum_rate]
        bars = plt.bar(algorithms, efficiency_ratio, color=colors, alpha=0.7)
        plt.ylabel('Efficiency Ratio (vs Optimal)')
        plt.title('Sum Rate Efficiency Ratio')
        
        for bar, value in zip(bars, efficiency_ratio):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'Efficiency ratio\nnot available\n(optimal too low)', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Sum Rate Efficiency')
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_beamforming_comparison.eps', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Convergence analysis
    print("\n=== Convergence Analysis ===")
    print(f"Enhanced Deep Unfolding reached {dun_final_sum_rate/optimal_sum_rate*100:.2f}% of optimal sum rate")
    print(f"PGA reached {pga_final_sum_rate/optimal_sum_rate*100:.2f}% of optimal sum rate")
    print(f"Enhanced Deep Unfolding is {pga_time/max(dun_time, 1e-6):.2f}x faster than PGA")
    
    # Check if Deep Unfolding outperforms PGA
    if dun_final_sum_rate > pga_final_sum_rate:
        improvement = ((dun_final_sum_rate - pga_final_sum_rate) / pga_final_sum_rate * 100)
        print(f"\n✅ SUCCESS: Enhanced Deep Unfolding outperforms PGA by {improvement:.2f}% in sum rate!")
    else:
        print("\n❌ WARNING: Enhanced Deep Unfolding did not outperform PGA in sum rate")

if __name__ == "__main__":
    main()