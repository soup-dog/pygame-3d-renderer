# Pygame CPU 3D Renderer

demo (125 cubes)
![](demo.mp4)

# resources consulted

https://gabrielgambetta.com/computer-graphics-from-scratch/11-clipping.html
https://gabrielgambetta.com/computer-graphics-from-scratch/09-perspective-projection.html
https://gabrielgambetta.com/computer-graphics-from-scratch/12-hidden-surface-removal.html
https://gabrielgambetta.com/computer-graphics-from-scratch/08-shaded-triangles.html

https://paroj.github.io/gltut/Positioning/Tut08%20Quaternions.html
https://paroj.github.io/gltut/Positioning/Tut06%20Rotation.html
https://paroj.github.io/gltut/Positioning/Tutorial%2007.html

https://www.khronos.org/opengl/wiki/Face_Culling

https://stackoverflow.com/questions/14905454/which-stage-of-pipeline-should-i-do-culling-and-clipping-and-how-should-i-recons

https://learnopengl.com/Getting-started/Coordinate-Systems
https://learnopengl.com/Advanced-OpenGL/Face-culling

https://mathworld.wolfram.com/Plane.html

https://gamedev.stackexchange.com/questions/120338/what-does-a-perspective-projection-matrix-look-like-in-opengl
https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix.html

# optimisations
- pre-padding - 25%
- pygame.gfxdraw vs pygame.draw - 50%
- np.linalg.norm vs handwritten magnitude (0.520 vs 0.335 per call)
- np.dot vs handwritten dot (0.125 vs 0.107 per call)
- np.cross vs handwritten cross (3.172 vs 0.2843 per call)

# TODO
- ~~painter's algorithm~~
- ~~non-square aspect ratio (need fix for frustum + perspective matrix)~~
- numba compilation
- ~~software rendered triangles using numba + direct access to pixels~~
- ~~z-buffer + textured triangles if fast enough~~
- enable software rendered triangles in renderer with different render modes
