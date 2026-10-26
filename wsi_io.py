"""Tile reader for the HuBMAP TIFFs: tiled RGB (YXS), tiled planes (CYX) and planar single-strip files."""
import math

import numpy as np
import tifffile


class WSI:
    def __init__(self, path):
        self.path = path
        self.tif = tifffile.TiffFile(path)
        s = self.tif.series[0]
        self.axes = s.axes
        p0 = self.tif.pages[0]
        if not p0.is_tiled:
            arr = tifffile.memmap(path)
            self._mm = arr.reshape(arr.shape[-3:])  # (C, H, W)
            self.height, self.width = self._mm.shape[1:]
            self.mode = "memmap"
        elif self.axes == "YXS":
            self.height, self.width = s.shape[:2]
            self.pages = [p0]
            self.mode = "rgb"
        elif self.axes == "CYX":
            self.height, self.width = s.shape[1:]
            self.pages = [self.tif.pages[i].aspage() for i in range(3)]
            self.mode = "planes"
        else:
            raise ValueError(f"unsupported layout {self.axes}")

    def close(self):
        self.tif.close()

    def _tile(self, page, idx):
        fh = page.parent.filehandle
        fh.seek(page.dataoffsets[idx])
        data = fh.read(page.databytecounts[idx])
        tile, _, _ = page.decode(data, idx, jpegtables=page.jpegtables)
        tile = np.asarray(tile)
        return tile.reshape(tile.shape[-3:])  # (th, tw, S)

    def iter_tiles(self, page):
        th, tw = page.tilelength, page.tilewidth
        ny, nx = math.ceil(self.height / th), math.ceil(self.width / tw)
        for ty in range(ny):
            for tx in range(nx):
                yield ty * th, tx * tw, self._tile(page, ty * nx + tx)

    def read_region(self, y0, x0, h, w):
        """Return an RGB uint8 array (h, w, 3); out-of-bounds areas are white."""
        out = np.full((h, w, 3), 255, np.uint8)
        ya, xa = max(y0, 0), max(x0, 0)
        yb, xb = min(y0 + h, self.height), min(x0 + w, self.width)
        if ya >= yb or xa >= xb:
            return out
        if self.mode == "memmap":
            out[ya - y0:yb - y0, xa - x0:xb - x0] = np.moveaxis(self._mm[:, ya:yb, xa:xb], 0, -1)
            return out
        for c, page in enumerate(self.pages):
            th, tw = page.tilelength, page.tilewidth
            nx = math.ceil(self.width / tw)
            for ty in range(ya // th, (yb - 1) // th + 1):
                for tx in range(xa // tw, (xb - 1) // tw + 1):
                    t = self._tile(page, ty * nx + tx)
                    ty0, tx0 = ty * th, tx * tw
                    sy0, sy1 = max(ya, ty0), min(yb, ty0 + th)
                    sx0, sx1 = max(xa, tx0), min(xb, tx0 + tw)
                    patch = t[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]
                    dst = out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
                    if self.mode == "rgb":
                        dst[:] = patch[..., :3]
                    else:
                        dst[..., c] = patch[..., 0]
        return out

    def thumbnail(self, step):
        """Nearest-neighbour thumbnail, sampling every `step` pixels."""
        H, W = math.ceil(self.height / step), math.ceil(self.width / step)
        if self.mode == "memmap":
            return np.ascontiguousarray(np.moveaxis(self._mm[:, ::step, ::step], 0, -1))
        out = np.full((H, W, 3), 255, np.uint8)
        for c, page in enumerate(self.pages):
            for y0, x0, t in self.iter_tiles(page):
                oy, ox = (-y0) % step, (-x0) % step
                sub = t[oy::step, ox::step]
                sy, sx = (y0 + oy) // step, (x0 + ox) // step
                sub = sub[:H - sy, :W - sx]
                if self.mode == "rgb":
                    out[sy:sy + sub.shape[0], sx:sx + sub.shape[1]] = sub[..., :3]
                else:
                    out[sy:sy + sub.shape[0], sx:sx + sub.shape[1], c] = sub[..., 0]
        return out
