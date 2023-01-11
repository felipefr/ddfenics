h = 20.0;
l = 50.0;
t = 10.0;
lc = 5.0;

p12 = newp;
Point(p12) = {0.0, 0.0, 0.0, lc};
p13 = newp;
Point(p13) = {h, 0.0, 0.0, lc};
p14 = newp;
Point(p14) = {h, l, 0.0 , lc};
p15 = newp;
Point(p15) = {0.0, l, 0.0 , lc};

p16 = newp;
Point(p16) = {0.0, 0.0, t , lc};
p17 = newp;
Point(p17) = {h, 0.0, t , lc};
p18 = newp;
Point(p18) = {h, l, t , lc};
p19 = newp;
Point(p19) = {0.0, l, t , lc};


l12 = newl;
Line(l12) = {p12, p13};
l13 = newl;
Line(l13) = {p13, p14};
l14 = newl;
Line(l14) = {p14, p15};
l15 = newl;
Line(l15) = {p15, p12};

//+
Line(5) = {2, 6};
//+
Line(6) = {1, 5};
//+
Line(7) = {6, 7};
//+
Line(8) = {3, 7};
//+
Line(9) = {5, 8};
//+
Line(10) = {4, 8};
//+
Line(11) = {7, 8};
//+
Line(12) = {6, 5};


//+
Curve Loop(1) = {12, -6, 1, 5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {8, 11, -10, -3};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {1, 2, 3, 4};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {7, 11, -9, -12};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {6, 9, -10, 4};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {2, 8, -7, -5};
//+
Plane Surface(6) = {6};
//+
Surface Loop(1) = {6, 3, 1, 4, 2, 5};
//+
Volume(1) = {1};



Physical Volume(0) = {1};
Physical Surface(1) = {1}; // left (y-)
Physical Surface(2) = {2}; // right (y+)
Physical Surface(3) = {3}; // bottom (z-)
Physical Surface(4) = {4}; // top (z+)
Physical Surface(5) = {5}; // back (x-)
Physical Surface(6) = {6}; // front (x+)

