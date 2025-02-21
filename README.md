<div> 
    <h1 align='center'> ⎏ GPU simulator ⎐</h1>
</div>


A GPU simulator project built in Python that mimics the core functionality of a GPU to perform dot products and matrix multiplications. This project demonstrates parallel computation by simulating Streaming Multiprocessors (SMs), Warps, and ALUs (Arithmetic Logic Units) to distribute, compute, and aggregate results. `The idea behind this is to be as supplementary media for a youtube video` I made for fun on the role GPUs play in machine learning to help others understand how they work and why fundamentally they have been such game changers. If you want to clone and create your own version and add to this, then please by all means do!

<div> 
    <h2 align='center'> Overview</h2>
</div>

----------

This project simulates a GPU architecture using Python and NumPy. It splits input data (matrices or vectors) across multiple simulated SMs. Each SM contains Warps that, in turn, contain ALUs that perform elementary arithmetic operations. The GPU simulator supports:
- **Dot Product:** Computes the dot product of two vectors.
- **Matrix Multiplication:** Performs matrix multiplication by executing element-wise multiplications and aggregating the results.

<div> 
    <h2 align='center'> Features</h2>
</div>

----------

- **GPU-like Architecture:** Simulated SMs, Warps, and ALUs.
- **Parallel Processing:** Distributes tasks among multiple simulated processing units.
- **Configurable Parameters:** Customize the number of SMs, warps per SM, threads per warp, and bit precision.
- **Supports Multiple Operations:** Dot product and matrix multiplication.

<div> 
    <h2 align='center'> Instance running</h2>
</div>

-------------

1. **Clone the Repository (install numpy if you don't have it):**
   ```bash
   git clone https://github.com/yourusername/GPU_dot_product.git
   cd GPU_dot_product
   pip install numpy

2. **Set up a virtual env (not needed but I like to do that, for python managment):**
   - [Anaconda Docs](https://docs.anaconda.com/anacondaorg/user-guide/)


<div> 
    <h2 align='center'> Testing</h2>
</div>

-----------

All of the testing I did is in a file, `instance.py` but you can literally just test it however you want as long as you correctly call the GPU_SIM constructor and have a reference

<div> 
    <h2 align='center'> Configuration</h2>
</div>

--------------

You can adjust the simulated GPU parameters by modifying the instance creation in `instance.py`:

- `num_sms`: Number of Streaming Multiprocessors.
- `num_warps`: Number of warps per SM.
- `num_threads_per_warp`: Number of ALU threads per warp.
- `num_bits`: Bit precision for ALU operations.

For matrix multiplication, ensure your SM configuration can handle all the tasks (e.g., adjust `num_warps` and `num_threads_per_warp` accordingly).

<div> 
    <h2 align='center'> File Structure</h2>
</div>

--------------
    
    GPU_dot_product/
    ├── gpu/
    │   ├── gpu_sim.py        # Main GPU simulator logic
    │   ├── sm.py             # Streaming Multiprocesso
    │   ├── warp.py           # Warp
    ├── logic_gates/
    │   ├── alu.py            # Arithmetic Logic Unit implementation
    │   ├── control.py        # ALU control logic
    │   ├── multiplexer.py    # Multiplexer implementation
    │   ├── ripple_adder.py   # Ripple-carry adder implementation
    ├── instance.py           # Example usage and testing script
    └── README.md             


