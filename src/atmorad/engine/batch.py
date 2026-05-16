import numpy as np

class PhotonBatch:
    def __init__(self, num_photons):
        self.pos = np.empty((3, num_photons), dtype=float)
        self.direction = np.empty((3, num_photons), dtype=float)
        self.tau = np.empty(num_photons, dtype=float)
        self.is_active = np.ones(num_photons, dtype=bool)
        
        self.ids = np.arange(num_photons)
        self.medium_ids = np.empty(num_photons, dtype=int)
        self.scatter_counts = np.zeros(num_photons, dtype=int)
    
    
    @property
    def active_count(self):
        return self.is_active.sum()
    
    def deactivate_photons(self, mask):
        self.is_active[mask] = False
        
    def shrink_to_active(self):
        self.ids = self.ids[self.is_active]
        self.pos = self.pos[:, self.is_active]
        self.direction = self.direction[:, self.is_active]
        self.tau = self.tau[self.is_active]
        self.medium_ids = self.medium_ids[self.is_active]
        self.scatter_counts = self.scatter_counts[self.is_active]
        self.is_active = self.is_active[self.is_active]