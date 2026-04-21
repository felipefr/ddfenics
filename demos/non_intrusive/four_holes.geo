// This code was created by pygmsh v6.0.2.
lc1 = 2.0;
lc2 = 1.0;

L = 100;
H = 80;
c1x = 25;
c1y = 40;
c2x = 75;
c2y = 60;
c3x = 50;
c3y = 20;
c4x = 75;
c4y = 20;
R1 = 15;
R2 = 15;
R3 = 7.5;
R4 = 7.5;
alpha1 = 0*Pi/180;
alpha2 = 30*Pi/180;
alpha3 = 50*Pi/180;
alpha4 = 50*Pi/180;
e1 = 1.0;
e2 = 1.0;
e3 = 1.0;
e4 = 1.0;

Point(1) = {0.0, 0.0, 0.0, lc1};
Point(2) = {L, 0.0, 0.0, lc1};
Point(3) = {L, H, 0.0, lc1};
Point(4) = {0., H, 0.0, lc1};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};



p0 = 4;
l0 = 4;
c0x = c1x; c0y = c1y; alpha0 = alpha1; R0=R1; e0=e1;
Point(p0+1) = {c0x, c0y, 0.0, lc2};
Point(p0+2) = {c0x + Cos(alpha0)*R0, c0y + Sin(alpha0)*R0, 0.0, lc2};
Point(p0+3) = {c0x - Cos(alpha0)*R0, c0y - Sin(alpha0)*R0, 0.0, lc2};
Point(p0+4) = {c0x - e0*Sin(alpha0)*R0, c0y + e0*Cos(alpha0)*R0, 0.0, lc2};
Point(p0+5) = {c0x + e0*Sin(alpha0)*R0, c0y - e0*Cos(alpha0)*R0, 0.0, lc2};

Ellipse(l0+1) = {p0+5, p0+1, p0+4, p0+2};
Ellipse(l0+2) = {p0+2, p0+1, p0+3, p0+4};
Ellipse(l0+3) = {p0+4, p0+1, p0+5, p0+3};
Ellipse(l0+4) = {p0+3, p0+1, p0+2, p0+5};


p0 = 9;
l0 = 8;
c0x = c2x; c0y = c2y; alpha0 = alpha2; R0=R2; e0=e2;
Point(p0+1) = {c0x, c0y, 0.0, lc2};
Point(p0+2) = {c0x + Cos(alpha0)*R0, c0y + Sin(alpha0)*R0, 0.0, lc2};
Point(p0+3) = {c0x - Cos(alpha0)*R0, c0y - Sin(alpha0)*R0, 0.0, lc2};
Point(p0+4) = {c0x - e0*Sin(alpha0)*R0, c0y + e0*Cos(alpha0)*R0, 0.0, lc2};
Point(p0+5) = {c0x + e0*Sin(alpha0)*R0, c0y - e0*Cos(alpha0)*R0, 0.0, lc2};

Ellipse(l0+1) = {p0+5, p0+1, p0+4, p0+2};
Ellipse(l0+2) = {p0+2, p0+1, p0+3, p0+4};
Ellipse(l0+3) = {p0+4, p0+1, p0+5, p0+3};
Ellipse(l0+4) = {p0+3, p0+1, p0+2, p0+5};


p0 = 14;
l0 = 12;
c0x = c3x; c0y = c3y; alpha0 = alpha3; R0=R3; e0=e3;
Point(p0+1) = {c0x, c0y, 0.0, lc2};
Point(p0+2) = {c0x + Cos(alpha0)*R0, c0y + Sin(alpha0)*R0, 0.0, lc2};
Point(p0+3) = {c0x - Cos(alpha0)*R0, c0y - Sin(alpha0)*R0, 0.0, lc2};
Point(p0+4) = {c0x - e0*Sin(alpha0)*R0, c0y + e0*Cos(alpha0)*R0, 0.0, lc2};
Point(p0+5) = {c0x + e0*Sin(alpha0)*R0, c0y - e0*Cos(alpha0)*R0, 0.0, lc2};

Ellipse(l0+1) = {p0+5, p0+1, p0+4, p0+2};
Ellipse(l0+2) = {p0+2, p0+1, p0+3, p0+4};
Ellipse(l0+3) = {p0+4, p0+1, p0+5, p0+3};
Ellipse(l0+4) = {p0+3, p0+1, p0+2, p0+5};


p0 = 19;
l0 = 16;
c0x = c4x; c0y = c4y; alpha0 = alpha4; R0=R4; e0=e4;
Point(p0+1) = {c0x, c0y, 0.0, lc2};
Point(p0+2) = {c0x + Cos(alpha0)*R0, c0y + Sin(alpha0)*R0, 0.0, lc2};
Point(p0+3) = {c0x - Cos(alpha0)*R0, c0y - Sin(alpha0)*R0, 0.0, lc2};
Point(p0+4) = {c0x - e0*Sin(alpha0)*R0, c0y + e0*Cos(alpha0)*R0, 0.0, lc2};
Point(p0+5) = {c0x + e0*Sin(alpha0)*R0, c0y - e0*Cos(alpha0)*R0, 0.0, lc2};

Ellipse(l0+1) = {p0+5, p0+1, p0+4, p0+2};
Ellipse(l0+2) = {p0+2, p0+1, p0+3, p0+4};
Ellipse(l0+3) = {p0+4, p0+1, p0+5, p0+3};
Ellipse(l0+4) = {p0+3, p0+1, p0+2, p0+5};


Curve Loop(1) = {4, 1, 2, 3};
Curve Loop(2) = {8, 5, 6, 7};
Curve Loop(3) = {15, 16, 13, 14};
Curve Loop(4) = {20, 17, 18, 19};
Curve Loop(5) = {12, 9, 10, 11};
Plane Surface(1) = {1, 2, 3, 4, 5};

Physical Surface("OMEGA", 1) = {1};
Physical Line("BOTTOM", 1) = {1};
Physical Line("RIGHT", 2) = {2};
Physical Line("TOP", 3) = {3};
Physical Line("LEFT", 4) = {4};

