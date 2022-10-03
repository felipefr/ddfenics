from fetricks.fenics.mesh.wrapper_gmsh import Gmsh
import numpy as np


class ellipsoidalInclusionsMesh(Gmsh):
    # def __init__(self, Lx = 1.0,Ly= 1.0, ellipseData = [], lcar = 0.05, ifPeriodic = False):
    def __init__(self, ellipseData, x0, y0, Lx, Ly , lcar):
        super().__init__()    
        
        self.x0 = x0
        self.y0 = y0
        self.Lx = Lx
        self.Ly = Ly
        self.lcar = lcar    
        self.eList = self.createEllipses(ellipseData,lcar)
        self.createSurfaces()
        self.physicalNaming()

    def createSurfaces(self):
        self.rec = self.add_rectangle(self.x0,self.x0 + self.Lx,self.y0,self.y0 + self.Ly, 0.0, lcar=self.lcar, holes = self.eList)
    
    def physicalNaming(self):
        self.add_physical(self.rec.surface, 1)
        self.add_physical(self.eList[:],0)
        self.add_physical(self.rec.lines,2)                
        
    def createEllipses(self, ellipseData, lcar):
        eList = []
        
        angles = [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]
        for cx, cy, l, e, t in ellipseData: # center, major axis length, excentricity, theta
            lenghts = [l,e*l,l,e*l]
            pc = self.add_point([cx,cy,0.0], lcar = lcar)
            pi =  [ self.add_point([cx + li*np.cos(ti + t), cy + li*np.sin(ti + t), 0.0], lcar = lcar) for li, ti in zip(lenghts,angles)]
            ai = [self.add_ellipse_arc(pi[i],pc,pi[i], pi[(i+1)%4]) for i in range(4)] # start, center, major axis, end
            a = self.add_line_loop(lines = ai)
            eList.append(self.add_surface(a))
        
        return eList
    
    def setTransfiniteBoundary(self,n):
        self.set_transfinite_lines(self.rec.lines, n)
        
