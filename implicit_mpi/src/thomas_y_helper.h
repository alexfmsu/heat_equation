#ifndef MPI_Helper_H
#define MPI_Helper_H

#include <mpi.h>

class Helper{

public:
    Helper(int _mpirank, int _mpisize, int _nx, int _my_ny, int _i_0){
        // -------------------------------------
        mpirank = _mpirank;
        mpisize = _mpisize;
        // -------------------------------------
        nx = _nx;
        my_ny = _my_ny;
        // -------------------------------------
        M = my_ny-1;
        
        H = mpisize;
        W = H + 1;
        // -------------------------------------
        i_0 = _i_0;
        i_down = M-1;
        // -------------------------------------
        req_diags_rcv = new MPI_Request[mpisize*3];
        req_diags_snd = new MPI_Request[mpisize*3];
        
        stat_recv = new MPI_Status[mpisize*3];
        // -------------------------------------
        req_sln_rcv = new MPI_Request[mpisize];
        req_sln_snd = new MPI_Request[mpisize];
        
        stat_sln = new MPI_Status[mpisize];
        // -------------------------------------
        B = new double[H];
        
        X = new double[H];
        // -------------------------------------
        triag_snd_buf = new double[nx+2];
        triple_dn = new double[nx+2];
        // -------------------------------------
        d_l = new double*[nx];
        d_r = new double*[nx];
        
        for(int i = 0; i < nx; i++){
            d_l[i] = new double[my_ny-1];
            d_r[i] = new double[my_ny-1];
        }
        // -------------------------------------
        block_size = new int[mpisize];
        
        block_start = new int[mpisize];
        block_fin = new int[mpisize];
        
        block_shift = new int[mpisize];
        
        for(int r = 0; r < mpisize; r++){
            block_size[r] = nx / mpisize;    
        }
        
        block_size[mpisize-1] += nx % mpisize;
        
        block_start[0] = 1;
        
        for(int r = 1; r < mpisize; r++){
            block_start[r] = 0;
            
            for(int i = 0; i < r; i++){
                block_start[r] += block_size[i];
            }
        }
        
        block_fin[0] = block_size[0];
        
        for(int r = 1; r < mpisize; r++){
            block_fin[r] = block_start[r] + block_size[r];
        }
        
        block_shift[0] = 0;
        
        for(int r = 1; r < mpisize; r++){
            block_shift[r] = block_start[r];
        }
        
        block_size_max = 0;
        
        for(int r = 0; r < mpisize; r++){
            block_size_max = std::max(block_size_max, block_size[r]);
        }
        // ---------------------------
        a_down_all = new double[nx-1];
        c_down_all = new double[nx-1];
        f_down_all = new double[nx-1];
        // ---------------------------
    }
    
    ~Helper(){
        // --------------------
        delete[] req_diags_rcv;
        delete[] req_diags_snd;
        
        delete[] stat_recv;
        // --------------------
        delete[] req_sln_rcv;
        delete[] req_sln_snd;
        
        delete[] stat_sln;
        // --------------------
        delete[] triag_snd_buf;
        delete[] triple_dn;
        // --------------------
        for(int i = 0; i < nx; i++){
            delete[] d_l[i];
            delete[] d_r[i];
        }
        
        delete[] d_l;
        delete[] d_r;
        // -------------------- 
        delete[] block_size;
        
        delete[] block_start;
        delete[] block_fin;
        
        delete[] block_shift;
        // -------------------- 
        delete[] a_down_all;
        delete[] c_down_all;
        delete[] f_down_all;
        // -------------------- 
    }
    
    void flush(){
        for(int J = 1; J < nx; J++){
            for(int i = 0; i < M; i++){
                d_l[J][i] = d_r[J][i] = 0;
            }
        }
    }
    
    // -----------------------
    int nx;
    int my_ny;
    // -----------------------
    int M;
    
    int H;
    int W;
    // -----------------------
    int i_0;
    int i_down;
    // -----------------------
    double** A;
    double* B;
    double** C;
    
    double** F;
    
    double* X;
    // -----------------------
    MPI_Request* req_diags_snd;
    MPI_Request* req_diags_rcv;
    
    MPI_Status* stat_recv;
    // -----------------------
    MPI_Request* req_sln_rcv;
    MPI_Request* req_sln_snd;
    
    MPI_Status* stat_sln;
    // -----------------------
    double* triag_snd_buf;
    double* triple_dn;
    // -----------------------
    double** d_l;
    double** d_r;
    // -----------------------
    MPI_Request req_triple_down_rcv;
    MPI_Request req_triags_snd;
    
    MPI_Status stat_triple_down;
    // -----------------------
    int* block_size;
    
    int* block_start;
    int* block_fin;
    
    int* block_shift;
    
    int block_size_max;
    // -----------------------
    double* a_down_all;
    double* c_down_all;
    double* f_down_all;
    // -----------------------
    
private:
    int mpirank;
    int mpisize;
};

#endif
