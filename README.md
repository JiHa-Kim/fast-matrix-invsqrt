== SPD size 256x256 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 ==
-- case gaussian_spd --
NS3           1.094 ms | resid 5.200e-03 | relerr 2.578e-03 | bad 0
NS4           1.018 ms | resid 5.187e-03 | relerr 2.563e-03 | bad 0
PE-NS3        1.122 ms | resid 7.604e-03 | relerr 3.842e-03 | bad 0
PE2           0.926 ms | resid 9.568e-03 | relerr 4.759e-03 | bad 0
AUTO          1.001 ms | resid 7.604e-03 | relerr 3.842e-03 | bad 0
BEST<=target(0.01): PE2 @ 0.926 ms, resid=9.568e-03

-- case illcond_1e6 --
NS3           0.853 ms | resid 4.470e-03 | relerr 2.233e-03 | bad 0
NS4           1.160 ms | resid 4.477e-03 | relerr 2.232e-03 | bad 0
PE-NS3        1.172 ms | resid 5.893e-03 | relerr 2.919e-03 | bad 0
PE2           1.005 ms | resid 1.095e-02 | relerr 5.473e-03 | bad 0
AUTO          1.126 ms | resid 5.893e-03 | relerr 2.919e-03 | bad 0
BEST<=target(0.01): NS3 @ 0.853 ms, resid=4.470e-03

-- case illcond_1e12 --
NS3           1.131 ms | resid 8.308e-03 | relerr 4.177e-03 | bad 0
NS4           1.788 ms | resid 8.316e-03 | relerr 4.173e-03 | bad 0
PE-NS3        1.111 ms | resid 9.017e-03 | relerr 4.539e-03 | bad 0
PE2           1.074 ms | resid 9.432e-03 | relerr 4.699e-03 | bad 0
AUTO          1.136 ms | resid 9.017e-03 | relerr 4.539e-03 | bad 0
BEST<=target(0.01): PE2 @ 1.074 ms, resid=9.432e-03

-- case near_rank_def --
NS3           0.850 ms | resid 7.875e-03 | relerr 3.944e-03 | bad 0
NS4           1.009 ms | resid 7.872e-03 | relerr 3.941e-03 | bad 0
PE-NS3        1.151 ms | resid 1.022e-02 | relerr 5.163e-03 | bad 0
PE2           0.981 ms | resid 8.031e-03 | relerr 4.004e-03 | bad 0
AUTO          0.993 ms | resid 1.022e-02 | relerr 5.163e-03 | bad 0
BEST<=target(0.01): NS3 @ 0.850 ms, resid=7.875e-03

-- case spike --
NS3           0.947 ms | resid 7.056e-03 | relerr 3.531e-03 | bad 0
NS4           1.194 ms | resid 7.041e-03 | relerr 3.528e-03 | bad 0
PE-NS3        1.090 ms | resid 6.155e-03 | relerr 3.084e-03 | bad 0
PE2           0.998 ms | resid 3.167e-03 | relerr 1.570e-03 | bad 0
AUTO          1.034 ms | resid 6.155e-03 | relerr 3.084e-03 | bad 0
BEST<=target(0.01): NS3 @ 0.947 ms, resid=7.056e-03

== SPD size 512x512 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 ==
-- case gaussian_spd --
NS3           0.945 ms | resid 4.242e-03 | relerr 2.188e-03 | bad 0
NS4           1.181 ms | resid 4.230e-03 | relerr 2.179e-03 | bad 0
PE-NS3        1.150 ms | resid 5.975e-03 | relerr 3.041e-03 | bad 0
PE2           0.927 ms | resid 5.594e-03 | relerr 2.881e-03 | bad 0
AUTO          1.193 ms | resid 5.975e-03 | relerr 3.041e-03 | bad 0
BEST<=target(0.01): PE2 @ 0.927 ms, resid=5.594e-03

-- case illcond_1e6 --
NS3           0.900 ms | resid 6.133e-03 | relerr 3.068e-03 | bad 0
NS4           1.227 ms | resid 6.132e-03 | relerr 3.065e-03 | bad 0
PE-NS3        1.132 ms | resid 6.926e-03 | relerr 3.440e-03 | bad 0
PE2           0.976 ms | resid 6.148e-03 | relerr 3.077e-03 | bad 0
AUTO          1.144 ms | resid 6.926e-03 | relerr 3.440e-03 | bad 0
BEST<=target(0.01): NS3 @ 0.900 ms, resid=6.133e-03

-- case illcond_1e12 --
NS3           0.941 ms | resid 6.380e-03 | relerr 3.133e-03 | bad 0
NS4           1.156 ms | resid 6.383e-03 | relerr 3.132e-03 | bad 0
PE-NS3        1.102 ms | resid 4.627e-03 | relerr 2.255e-03 | bad 0
PE2           1.139 ms | resid 3.858e-03 | relerr 1.894e-03 | bad 0
AUTO          1.102 ms | resid 4.627e-03 | relerr 2.255e-03 | bad 0
BEST<=target(0.01): NS3 @ 0.941 ms, resid=6.380e-03

-- case near_rank_def --
NS3           0.931 ms | resid 7.820e-03 | relerr 3.862e-03 | bad 0
NS4           1.159 ms | resid 7.820e-03 | relerr 3.860e-03 | bad 0
PE-NS3        1.090 ms | resid 7.806e-03 | relerr 3.875e-03 | bad 0
PE2           1.058 ms | resid 7.550e-03 | relerr 3.721e-03 | bad 0
AUTO          1.087 ms | resid 7.806e-03 | relerr 3.875e-03 | bad 0
BEST<=target(0.01): NS3 @ 0.931 ms, resid=7.820e-03

-- case spike --
NS3           0.970 ms | resid 7.672e-03 | relerr 3.812e-03 | bad 0
NS4           1.152 ms | resid 7.637e-03 | relerr 3.798e-03 | bad 0
PE-NS3        1.083 ms | resid 6.827e-03 | relerr 3.399e-03 | bad 0
PE2           0.968 ms | resid 3.623e-03 | relerr 1.853e-03 | bad 0
AUTO          1.091 ms | resid 6.827e-03 | relerr 3.399e-03 | bad 0
BEST<=target(0.01): PE2 @ 0.968 ms, resid=3.623e-03

== SPD size 1024x1024 | dtype=torch.bfloat16 | compile=False | precond=aol | l_target=0.05 ==
-- case gaussian_spd --
NS3           1.990 ms | resid 1.126e-02 | relerr 5.672e-03 | bad 0
NS4           2.526 ms | resid 1.132e-02 | relerr 5.668e-03 | bad 0
PE-NS3        2.287 ms | resid 3.204e-03 | relerr 1.581e-03 | bad 0
PE2           1.941 ms | resid 1.249e-03 | relerr 6.338e-04 | bad 0
AUTO          1.939 ms | resid 1.249e-03 | relerr 6.338e-04 | bad 0
BEST<=target(0.01): AUTO @ 1.939 ms, resid=1.249e-03

-- case illcond_1e6 --
NS3           1.946 ms | resid 1.092e-02 | relerr 5.605e-03 | bad 0
NS4           2.570 ms | resid 1.092e-02 | relerr 5.603e-03 | bad 0
PE-NS3        2.399 ms | resid 2.776e-03 | relerr 1.326e-03 | bad 0
PE2           1.957 ms | resid 6.727e-04 | relerr 3.338e-04 | bad 0
AUTO          1.912 ms | resid 6.727e-04 | relerr 3.338e-04 | bad 0
BEST<=target(0.01): AUTO @ 1.912 ms, resid=6.727e-04

-- case illcond_1e12 --
NS3           1.946 ms | resid 1.089e-02 | relerr 5.560e-03 | bad 0
NS4           2.526 ms | resid 1.089e-02 | relerr 5.558e-03 | bad 0
PE-NS3        2.244 ms | resid 2.385e-03 | relerr 1.121e-03 | bad 0
PE2           2.028 ms | resid 5.568e-04 | relerr 2.702e-04 | bad 0
AUTO          2.011 ms | resid 5.568e-04 | relerr 2.702e-04 | bad 0
BEST<=target(0.01): AUTO @ 2.011 ms, resid=5.568e-04

-- case near_rank_def --
NS3           1.938 ms | resid 1.087e-02 | relerr 5.534e-03 | bad 0
NS4           2.538 ms | resid 1.087e-02 | relerr 5.532e-03 | bad 0
PE-NS3        2.631 ms | resid 2.175e-03 | relerr 1.018e-03 | bad 0
PE2           1.996 ms | resid 5.563e-04 | relerr 2.684e-04 | bad 0
AUTO          1.940 ms | resid 5.563e-04 | relerr 2.684e-04 | bad 0
BEST<=target(0.01): AUTO @ 1.940 ms, resid=5.563e-04

-- case spike --
NS3           1.928 ms | resid 1.010e-02 | relerr 5.000e-03 | bad 0
NS4           2.508 ms | resid 1.008e-02 | relerr 4.994e-03 | bad 0
PE-NS3        2.332 ms | resid 1.794e-03 | relerr 8.611e-04 | bad 0
PE2           2.003 ms | resid 1.542e-03 | relerr 7.277e-04 | bad 0
AUTO          1.999 ms | resid 1.542e-03 | relerr 7.277e-04 | bad 0
BEST<=target(0.01): AUTO @ 1.999 ms, resid=1.542e-03
