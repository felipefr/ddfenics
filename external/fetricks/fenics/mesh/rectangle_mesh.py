from fetricks.fenics.mesh.wrapper_gmsh import myGmsh

class rectangleMesh(Gmsh):
    def __init__(self, x0, y0, Lx, Ly , lcar):
        super().__init__()
        
        self.x0 = x0
        self.y0 = y0
        self.Lx = Lx
        self.Ly = Ly
        self.lcar = lcar    
        self.createSurfaces()
        self.physicalNaming()

    def createSurfaces(self):
        self.rec = self.add_rectangle(self.x0,self.x0 + self.Lx,self.y0,self.y0 + self.Ly, 0.0, lcar=self.lcar)
    
    def physicalNaming(self):
        self.add_physical(self.rec.surface, 'vol')
        [self.add_physical(e,'side' + str(i)) for i, e in enumerate(self.rec.lines)]
    
    def setTransfiniteBoundary(self,n):
        self.set_transfinite_lines(self.rec.lines, n)
        

                


