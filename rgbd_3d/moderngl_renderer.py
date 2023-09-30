import os
import ctypes
from easydict import EasyDict as edict
import numpy as np
from tqdm import tqdm

import moderngl
import glm


class SimpleRenderer(object):
    """
    OpenGL renderer of 3D meshes using raw vertex colors with moderngl.

    Args:
        image_size (int): rendered image size
        near (float): near plane
        far (float): far plane
    """

    def __init__(self, render_size=128, image_size=128, near=0.01, far=200.0, device=0):
        self.render_size = render_size
        self.image_size = image_size
        self.near = near
        self.far = far
        self.mgl_ctx = None
        self.shader = None
        self.init_gl_context(device)
    
    def init_gl_context(self, device):
        """
        Initialize OpenGL context.
        """
        # init gl context
        ## Initialize EGL
        self.mgl_ctx = moderngl.create_context(standalone=True, backend='egl', device_index=device)

        # init glsl shaders
        ## load shaders
        with open(os.path.join(os.path.dirname(__file__), 'shaders', 'simple.vsh'), 'r') as f:
            vertex_shader = f.read()
        with open(os.path.join(os.path.dirname(__file__), 'shaders', 'simple.fsh'), 'r') as f:
            fragment_shader = f.read()
        ## compile shaders
        self.shader = self.mgl_ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self.shader['colortex'].value = 0

        # init gl states
        self.mgl_ctx.depth_func = '<'
        self.mgl_ctx.enable(moderngl.DEPTH_TEST)
        self.mgl_ctx.disable(moderngl.CULL_FACE)
        self.mgl_ctx.disable(moderngl.BLEND)
        self.mgl_ctx.front_face = 'ccw'

        # init gl buffers
        self.vbo = self.mgl_ctx.buffer(reserve=6 * 4 * (self.image_size + 2)**2, dynamic=True)
        self.ibo = self.mgl_ctx.buffer(reserve=6 * 4 * (self.image_size + 1)**2, dynamic=True)
        vao_content = [
            (self.vbo, '3f 2f 1f', 'i_position', 'i_texcoord', 'i_flag'),
        ]
        self.vao = self.mgl_ctx.vertex_array(self.shader, vao_content, self.ibo)

        # init gl textures
        self.color_fb = self.mgl_ctx.texture((self.render_size, self.render_size), 4, dtype='f4')
        self.depth_fb = self.mgl_ctx.depth_texture((self.render_size, self.render_size))
        self.color_fb.filter = moderngl.NEAREST, moderngl.NEAREST
        self.depth_fb.filter = moderngl.NEAREST, moderngl.NEAREST
        self.fbo = self.mgl_ctx.framebuffer(color_attachments=[self.color_fb], depth_attachment=self.depth_fb)

        # texture
        ## color
        self.color_texture = self.mgl_ctx.texture((self.image_size, self.image_size), 3, dtype='f4')
        self.color_texture.filter = moderngl.NEAREST, moderngl.NEAREST
        self.color_texture.repeat_x = False
        self.color_texture.repeat_y = False

    def __del__(self):
        """
        Destructor.
        """
        self.mgl_ctx.release()
        self.shader.release()
        self.vbo.release()
        self.ibo.release()
        self.vao.release()
        self.color_fb.release()
        self.depth_fb.release()
        self.fbo.release()
        self.color_texture.release()

    def render(self, mesh, color, modelview, fov=45.0):
        """
        Render a scene.

        Args:
            scene (Scene): scene to render

        Returns:
            np.array (H, W, 3): rendered color
            np.array (H, W, 3): rendered normal
            np.array (H, W, 1): rendered mask
        """
        # init
        self.fbo.use()
        self.fbo.viewport = (0, 0, self.render_size, self.render_size)

        # render
        self.mgl_ctx.clear(0.0, 0.0, 0.0, 0.0)
        projection = glm.perspective(glm.radians(fov), 1, self.near, self.far)
        self.shader['u_projection'].write(projection)
        vertices = np.concatenate([
            mesh.vertices.position,
            mesh.vertices.uv,
            mesh.vertices.flag,
        ], axis=-1).astype(np.float32)
        faces = mesh.faces.astype(np.uint32)
        self.vbo.write(vertices)
        self.ibo.write(faces)
        self.color_texture.write(color.astype(np.float32))
        self.color_texture.use(0)

        if not isinstance(modelview, list):
            modelview = [modelview]

        ret = []
        for mv in modelview:
            self.shader['u_modelview'].write(mv)
            self.vao.render(moderngl.TRIANGLES, vertices=faces.size)
        
            # read pixels
            pixels = np.frombuffer(self.color_fb.read(), dtype=np.float32).reshape(self.render_size, self.render_size, 4)
            pixels = np.flip(pixels, axis=0)
            color = pixels[:, :, :3]
            mask = pixels[:, :, 3:] > 0.5
            # read depth
            depth = np.frombuffer(self.depth_fb.read(), dtype=np.float32).reshape(self.render_size, self.render_size, 1)
            ## linearize depth
            depth = self.near * self.far / (self.far - depth * (self.far - self.near))
            depth = np.flip(depth, axis=0)
            depth = depth.astype(np.float32)

            ret.append(edict({
                'color': color,
                'depth': depth,
                'mask': mask,
            }))

        return ret if len(ret) > 1 else ret[0]


class AggregationRenderer(object):
    """
    OpenGL renderer of 3D meshes with aggregation using moderngl.

    Args:
        image_size (int): rendered image size
        near (float): near plane
        far (float): far plane
    """

    def __init__(self, render_size=128, image_size=128, near=0.01, far=200.0, device=0, max_views=27):
        self.render_size = render_size
        self.image_size = image_size
        self.near = near
        self.far = far
        self.mgl_ctx = None
        self.shader = None
        self.max_views = max_views
        self.init_gl_context(device)
    
    def init_gl_context(self, device):
        """
        Initialize OpenGL context.
        """
        # init gl context
        ## Initialize EGL
        self.mgl_ctx = moderngl.create_context(standalone=True, backend='egl', device_index=device)

        # init glsl shaders
        ## load shaders
        with open(os.path.join(os.path.dirname(__file__), 'shaders', 'aggregation.vsh'), 'r') as f:
            vertex_shader = f.read()
        with open(os.path.join(os.path.dirname(__file__), 'shaders', 'aggregation.fsh'), 'r') as f:
            fragment_shader = f.read()
        with open(os.path.join(os.path.dirname(__file__), 'shaders', 'clear.csh'), 'r') as f:
            clear_cs = f.read()
        with open(os.path.join(os.path.dirname(__file__), 'shaders', 'aggregation.csh'), 'r') as f:
            aggragation_cs = f.read()
        ## compile shaders
        self.shader = self.mgl_ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self.clear_cs = self.mgl_ctx.compute_shader(clear_cs)
        self.aggragation_cs = self.mgl_ctx.compute_shader(aggragation_cs)
        self.shader['colortex'].value = 0
        self.aggragation_cs['colortex'].value = 0
        self.aggragation_cs['depthtex'].value = 1

        # init gl states
        self.mgl_ctx.depth_func = '<'
        self.mgl_ctx.enable(moderngl.DEPTH_TEST)
        self.mgl_ctx.disable(moderngl.CULL_FACE)
        self.mgl_ctx.disable(moderngl.BLEND)
        self.mgl_ctx.front_face = 'ccw'

        # init gl buffers
        self.vbos = []
        self.ibos = []
        self.vaos = []
        self.color_texture = []
        for i in range(self.max_views):
            vbo = self.mgl_ctx.buffer(reserve=9 * 4 * (self.image_size + 2)**2, dynamic=True)
            ibo = self.mgl_ctx.buffer(reserve=6 * 4 * (self.image_size + 1)**2, dynamic=True)
            vao_content = [
                (vbo, '3f 3f 2f 1f', 'i_position', 'i_normal', 'i_texcoord', 'i_flag'),
            ]
            vao = self.mgl_ctx.vertex_array(self.shader, vao_content, ibo)
            color_texture = self.mgl_ctx.texture((self.image_size, self.image_size), 3, dtype='f4')
            color_texture.filter = moderngl.NEAREST, moderngl.NEAREST
            color_texture.repeat_x = False
            color_texture.repeat_y = False
            self.vbos.append(vbo)
            self.ibos.append(ibo)
            self.vaos.append(vao)
            self.color_texture.append(color_texture)

        # init gl textures
        self.color_fb = self.mgl_ctx.texture((self.render_size, self.render_size), 4, dtype='f4')
        self.depth_fb = self.mgl_ctx.depth_texture((self.render_size, self.render_size))
        self.color_fb.filter = moderngl.NEAREST, moderngl.NEAREST
        self.depth_fb.filter = moderngl.NEAREST, moderngl.NEAREST
        self.fbo = self.mgl_ctx.framebuffer(color_attachments=[self.color_fb], depth_attachment=self.depth_fb)
        ## aggregation texture
        self.color_agg_texture = self.mgl_ctx.texture((self.render_size, self.render_size), 4, dtype='f4')
        self.depth_agg_texture = self.mgl_ctx.texture((self.render_size, self.render_size), 2, dtype='f4')
        self.mask_agg_texture = self.mgl_ctx.texture((self.render_size, self.render_size), 2, dtype='f4')
        self.color_agg_texture.filter = moderngl.NEAREST, moderngl.NEAREST
        self.depth_agg_texture.filter = moderngl.NEAREST, moderngl.NEAREST
        ## bind compute shader buffers
        self.color_agg_texture.bind_to_image(0, read=True, write=True)
        self.depth_agg_texture.bind_to_image(1, read=True, write=True)
        self.mask_agg_texture.bind_to_image(2, read=True, write=True)

    def __del__(self):
        """
        Destructor.
        """
        self.mgl_ctx.release()
        self.shader.release()
        for i in range(self.max_views):
            self.vbos[i].release()
            self.ibos[i].release()
            self.vaos[i].release()
            self.color_texture[i].release()
        self.color_fb.release()
        self.depth_fb.release()
        self.fbo.release()
        self.color_agg_texture.release()
        self.depth_agg_texture.release()
        self.mask_agg_texture.release()

    def render(self, meshes, colors, modelview, fov=45.0, is_autoregressive=False, verbose=False, tqdm_args={}):
        """
        Render a scene.

        Args:
            meshes (list): meshes to render
            colors (list): colors of meshes
            modelview (glm.mat4): modelview matrix
            fov (float): field of view

        Returns:
            np.array (H, W, 3): rendered color
            np.array (H, W, 3): rendered normal
            np.array (H, W, 1): rendered mask
        """
        with self.mgl_ctx:
            # init
            self.fbo.use()
            self.fbo.viewport = (0, 0, self.render_size, self.render_size)

            # buffer objects
            for i, mesh in enumerate(meshes):
                if is_autoregressive and i != len(meshes) - 1:
                    continue
                vertices = np.concatenate([
                    mesh.vertices.position,
                    mesh.vertices.normal,
                    mesh.vertices.uv,
                    mesh.vertices.flag,
                ], axis=-1).astype(np.float32)
                faces = mesh.faces.astype(np.uint32)
                self.vbos[i].write(vertices)
                self.ibos[i].write(faces)
                self.color_texture[i].write(np.ascontiguousarray(colors[i].astype(np.float32)))

            # render
            projection = glm.perspective(glm.radians(fov), 1, self.near, self.far)
            self.shader['u_projection'].write(projection)

            if not isinstance(modelview, list):
                modelview = [modelview]
            ret = []
            for mv in tqdm(modelview, disable=not verbose, desc='rendering', **tqdm_args):
                self.shader['u_modelview'].write(mv)
                # c2w = glm.inverse(mv)
                # self.shader['u_camera'].write(glm.vec3(c2w[3]))
                self.clear_cs.run(group_x=self.render_size // 8, group_y=self.render_size // 8)
                for i in range(len(meshes)):
                    self.color_texture[i].use(0)
                    self.fbo.clear(0.0, 0.0, 0.0, 0.0)
                    c2w = glm.inverse(meshes[i].modelview)
                    self.shader['u_sample_camera'].write(glm.vec3(c2w[3]))
                    self.vaos[i].render(moderngl.TRIANGLES, vertices=meshes[i].faces.size)
                    self.color_fb.use(0)
                    self.depth_fb.use(1)
                    self.aggragation_cs.run(group_x=self.render_size // 8, group_y=self.render_size // 8)
            
                # read pixels
                pixels = np.frombuffer(self.color_agg_texture.read(), dtype=np.float32).reshape(self.render_size, self.render_size, 4)
                pixels = np.flip(pixels, axis=0)
                color = np.where(pixels[:, :, 3:] > 0.0, pixels[:, :, :3] / np.maximum(pixels[:, :, 3:], 1e-24), 0.0)
                # read depth
                depth = np.frombuffer(self.depth_agg_texture.read(), dtype=np.float32).reshape(self.render_size, self.render_size, 2)
                depth = np.flip(depth, axis=0)
                depth = np.where(depth[:, :, 1:] > 0.0, depth[:, :, :1] / np.maximum(depth[:, :, 1:], 1e-24), 0.0)
                depth = self.near * self.far / (self.far - depth * (self.far - self.near))
                depth = depth.astype(np.float32)
                # read mask
                mask = np.frombuffer(self.mask_agg_texture.read(), dtype=np.float32).reshape(self.render_size, self.render_size, 2)
                mask = np.flip(mask, axis=0)
                mask_color = mask[:, :, 1:] > 0.5
                mask_depth = mask[:, :, :1] > 0.5

                ret.append(edict({
                    'color': color,
                    'depth': depth,
                    'mask_color': mask_color,
                    'mask_depth': mask_depth,
                }))

            return ret if len(ret) > 1 else ret[0]


if __name__ == '__main__':
    import imageio
    import trimesh
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import cv2

    renderer = SimpleRenderer(
        render_size=128,
        image_size=2,
        near=0.1,
        far=200,
        device=0,
    )
    mesh = edict({
        'vertices': edict({
            'position': np.array([
                [-1, -1, 0],
                [1, -1, 0],
                [1, 1, 0],
                [-1, 1, 0],
            ], dtype=np.float32),
            'uv': np.array([
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
            ], dtype=np.float32),
            'flag': np.array([1, 1, 1, 1], dtype=np.float32),
        }),
        'faces': np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.uint32),
    })
    color = np.array([
        [[1, 0, 0], [0, 1, 0]],
        [[0, 0, 1], [1, 1, 0]],
    ], dtype=np.float32)

    modelview = glm.lookAt(
        glm.vec3(0.0, 0.0, 1.0),
        glm.vec3(0.0, 0.0, 0.0),
        glm.vec3(0.0, 1.0, 0.0)
    )
    
    res = renderer.render(mesh, color, modelview, fov=90.0)
    imageio.imwrite('dbg.png', (np.clip(res.color, 0, 1) * 255).astype(np.uint8))

