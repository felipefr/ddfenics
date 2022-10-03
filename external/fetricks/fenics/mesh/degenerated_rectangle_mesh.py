from fetricks.fenics.mesh.wrapper_gmsh import Gmsh

class degeneratedBoundaryRectangleMesh(Gmsh): # Used for implementation of L2bnd
    def __init__(self, x0, y0, Lx, Ly, Nb): 
        super().__init__()
        
        self.x0 = x0
        self.y0 = y0
        self.Lx = Lx
        self.Ly = Ly
        self.lcar = 4*self.Lx    ## just a huge value
        self.createSurfaces()
        self.physicalNaming()
        self.set_transfinite_lines(self.extLines, Nb)

    def createSurfaces(self):
        p1 = self.add_point([self.x0, self.y0, 0.0], lcar = self.lcar)
        p2 = self.add_point([self.x0 + self.Lx, self.y0, 0.0], lcar = self.lcar)
        p3 = self.add_point([self.x0 + self.Lx, self.y0 + self.Ly ,0.0], lcar = self.lcar)
        p4 = self.add_point([self.x0 , self.y0 + self.Ly ,0.0], lcar = self.lcar)
        p5 = self.add_point([self.x0 + 0.5*self.Lx, self.y0 + 0.5*self.Ly ,0.0], lcar = self.lcar)
        
        p = [p1,p2,p3,p4]
        
        self.extLines = [ self.add_line(p[i],p[(i+1)%4]) for i in range(4) ]
        self.intLines = [ self.add_line(p[i],p5) for i in range(4) ]
        
        LineLoops = [ self.add_line_loop(lines = [-self.intLines[i], self.extLines[i] ,self.intLines[(i+1)%4]]) for i in range(4)]
        self.Surfs = []
        for ll in LineLoops:
            self.Surfs.append(self.add_surface(ll))
    
    def physicalNaming(self):
        self.add_physical(self.Surfs, 'vol')
        [self.add_physical(e,'side' + str(i)) for i, e in enumerate(self.extLines)]