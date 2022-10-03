from dolfin import compile_cpp_code, UserExpression, CompiledExpression
import dolfin as df
from fetricks.fenics.la.wrapper_solvers import local_project

code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <dolfin/function/Expression.h>

typedef Eigen::VectorXd npArray;
typedef Eigen::VectorXi npArrayInt;

class myCoeff : public dolfin::Expression {
  public:
    
    npArray coeffs; // dynamic double vector
    npArrayInt materials; // dynamic integer vector
    int nc;
    
    myCoeff(npArray c, npArrayInt mat, int nc) : dolfin::Expression(nc), coeffs(c), materials(mat), nc(nc) { }

    void eval(Eigen::Ref<Eigen::VectorXd> values,
                      Eigen::Ref<const Eigen::VectorXd> x,
                      const ufc::cell& cell) const {
        
        int ii = nc*materials[cell.index];              
        for(int i = 0; i<nc ; i++) values[i] = coeffs[ii + i];
    }
    
   void updateCoeffs(Eigen::VectorXd c, int n){ 
       for(int i = 0; i<n; i++){
          coeffs[i]= c[i];
       }
   }
                      
                    
};

PYBIND11_MODULE(SIGNATURE, m) {
    pybind11::class_<myCoeff, std::shared_ptr<myCoeff>, dolfin::Expression>
    (m, "myCoeff")
    .def(pybind11::init<npArray,npArrayInt, int>())
    .def("__call__", &myCoeff::eval)
    .def("updateCoeffs", &myCoeff::updateCoeffs);
}
"""


class myCoeff(UserExpression):
    def __init__(self, markers, coeffs, **kwargs):
        self.markers = markers
        self.coeffs = coeffs
        if(len(coeffs.shape)>1):
            self.ncoeff = coeffs.shape[1]
        else:
            self.ncoeff = 1
            
        super().__init__(**kwargs)

        
    def eval_cell(self, values, x, cell):
        values[:self.ncoeff] = self.coeffs[self.markers[cell.index]]
        
    def value_shape(self):
        return (self.ncoeff,)
    

compCode = compile_cpp_code(code)
myCoeffCpp = lambda x,y : CompiledExpression(compCode.myCoeff(x.flatten().astype('float64'),y, x.shape[1]),degree = 0)

def getMyCoeff(materials, param, op = 'cpp', mesh = None): 
    if(op == 'cpp'):
        return myCoeffCpp(param,materials)
        
    elif(op == 'python'):
        return myCoeff(materials, param, degree = 2) # it was 0 before

    elif(op == 'function'):
        coeff_exp = myCoeffCpp(param,materials)
        
        V = df.VectorFunctionSpace(mesh, 'DG', 0, dim = len(param[0]))
        return local_project(coeff_exp, V)
        
        
        
        
        

    