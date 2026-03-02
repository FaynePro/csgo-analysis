import json
import pickle
from typing import Dict, Tuple, Optional

import numpy as np
from pyparsing import Path
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from utils.project_root import find_project_root

class TacticFeatureDataset(Dataset):
    def __init__(
        self,
        data_root_dir: str,
        label_to_id: Optional[Dict[str, int]] = None,
        feature_scaler: Optional[StandardScaler] = None,
        tactics_json_path: str = ""
    ):
        """
        Initialize the TacticFeatureDataset.
        
        Args:
            data_root_dir: Path to directory containing pickle files
            label_to_id: Pre-computed label to ID mapping (for inference)
            feature_scaler: Pre-fitted StandardScaler (for inference)
            tactics_json_path: Path to JSON file containing tactic definitions
        """
        super().__init__()
        project_root = find_project_root()
        self.data_root_dir = project_root / data_root_dir
        self.all_data = []
        
        self.all_data = self._load_data_from_directory()
        
        # If returning empty dataset, raise error
        if len(self.all_data) == 0:
            raise ValueError(f"No valid data found in {self.data_root_dir}")
        
        # Extract and normalize features
        self.all_features = self._collect_all_features()
        
        # Initialize or use provided feature scaler for normalization
        if feature_scaler is not None:
            self.feature_scaler = feature_scaler
        else:
            self.feature_scaler = StandardScaler()
            # Fit scaler on all collected features for proper normalization
            if len(self.all_features) > 0:
                self.feature_scaler.fit(self.all_features)
        
        # Initialize or use provided label mapping
        if label_to_id is not None:
            self.label_to_id = label_to_id
        else:
            tactics_full_root = project_root / tactics_json_path
            # Load tactic definitions from JSON and create label mapping
            with open(tactics_full_root) as f:
                tactics = json.load(f)
            strategies = [item["id"] for item in tactics]
            self.label_to_id = {label: idx for idx, label in enumerate(strategies)}
        
        # Process all data into feature vectors and labels
        self.processed_data = [
            self._process_data_to_features(data)
            for data, _, _ in self.all_data
        ]
    
    def _load_data_from_directory(self):
        """
        Recursively scan the directory tree for pickle files containing game data.
        Filters out frames that do not have valid strategy labels.

        Each data file is expected to contain a dictionary structured as:
            {
                0: [frame0_dict, frame1_dict, ...],  # round 0
                1: [frame0_dict, frame1_dict, ...],  # round 1
                ...
            }

        Each frame dictionary should have the following keys:
            - "round_data": dict containing round-specific info
            - "frame_data": dict containing frame-specific info
            - "players_data": list of player dictionaries
            - "bomb_data": dict with bomb-related info
            - "tactic": string representing the strategy used
            - "map_name": string representing the map name
        """
        all_data = []
        missing_count = 0
        non_missing_count = 0

        for file_path in Path(self.data_root_dir).rglob("*.pkl"):
            try:
                with file_path.open("rb") as f:
                    data_in_file = pickle.load(f)

                if not isinstance(data_in_file, dict):
                    continue

                for round_idx, frames in data_in_file.items():
                    for frame_idx, frame_data in enumerate(frames):
                        tactic = frame_data.get("tactic")
                        if tactic and tactic != "unknown":
                            all_data.append((frame_data, str(file_path), (round_idx, frame_idx)))
                            non_missing_count += 1
                        else:
                            missing_count += 1

            except (pickle.UnpicklingError, FileNotFoundError, EOFError) as e:
                print(f"Failed to load {file_path}: {e}")

        print(f"Total frames with tactic: {non_missing_count}")
        print(f"Total frames missing tactic: {missing_count}")

        return all_data
    
    def _extract_raw_node_features(self, data: dict, max_nodes: int) -> np.ndarray:
        """
        Extract raw node features from a single frame dictionary.
        
        Features are taken from `players_data` (one node per player) and optionally the bomb.
        
        Returns:
            np.ndarray of shape (max_nodes, node_feature_size)
        """
        # Example: use player features only for now
        player_nodes = data["players_data"]  # list of dicts
        node_features_list = []

        for player in player_nodes:
            # Flatten relevant player features into a vector
            # Adjust depending on what features you want to include
            features = [
                player["x"], player["y"], player["z"],
                player["velocityX"], player["velocityY"], player["velocityZ"],
                player["viewX"], player["viewY"],
                player["hp"], player["armor"],
                player["activeWeapon"],
                player["totalUtility"],
                player["isAlive"], player["isDefusing"], player["isPlanting"],
                player["isReloading"], player["isInBombZone"], player["isInBuyZone"],
                player["equipmentValue"], player["equipmentValueFreezetimeEnd"],
                player["equipmentValueRoundStart"],
                player["cash"], player["cashSpendThisRound"], player["cashSpendTotal"],
                player["hasHelmet"], player["hasDefuse"], player["hasBomb"]
            ]
            node_features_list.append(features)

        # Optional: include bomb as an extra node
        # bomb = data["bomb_data"]
        # node_features_list.append([bomb["x"], bomb["y"], bomb["z"], ...])

        # Pad with zeros if fewer nodes than max_nodes
        current_nodes = len(node_features_list)
        feature_size = len(node_features_list[0]) if node_features_list else 0
        padding_needed = max_nodes - current_nodes

        if padding_needed > 0:
            node_features_list.extend([[0.0] * feature_size] * padding_needed)

        return np.array(node_features_list, dtype=np.float32)  # shape: (max_nodes, feature_size)


    def _collect_all_features(self) -> np.ndarray:
        all_features = []
        max_nodes = 0

        # Flatten self.all_data if it is stored as dict of rounds
        flat_data = []
        if isinstance(self.all_data, dict):
            print("Flattening graphs from dict structure")
            for round_idx, frames in self.all_data.items():
                for frame_idx, frame_data in enumerate(frames):
                    flat_data.append((frame_data, round_idx, frame_idx))
        else:
            flat_data = self.all_data

        print(f"Total frames to process: {len(flat_data)}")

        # First pass: find maximum number of nodes in any frame
        for idx, (frame_data, round_idx, frame_idx) in enumerate(flat_data):
            num_nodes = len(frame_data["players_data"])
            max_nodes = max(max_nodes, num_nodes)
        self.max_nodes = max_nodes
        print(f"Maximum number of nodes across all frames: {max_nodes}")

        # Second pass: extract and pad node features
        for idx, (frame_data, round_idx, frame_idx) in enumerate(flat_data):
            node_features = self._extract_raw_node_features(frame_data, max_nodes)
            all_features.append(node_features.flatten())

        if all_features:
            result = np.array(all_features)
            print(f"Final feature array shape: {result.shape}")
            return result
        else:
            print("No features extracted; returning empty array")
            return np.array([])

    
    def _process_data_to_features(self, data_dict: Dict) -> Tuple[torch.Tensor, int]:
        """
        Convert a single frame to normalized raw node features and label.
        
        Args:
            data_dict: Frame data dictionary
            
        Returns:
            Tuple of (normalized_raw_features tensor, label_id)
        """
        # Extract raw node features and pad to max_nodes
        raw_features = self._extract_raw_node_features(data_dict, self.max_nodes)
        
        # Flatten features to 1D and normalize
        normalized_features = self.feature_scaler.transform(raw_features.flatten().reshape(1, -1))[0]
        features_tensor = torch.tensor(normalized_features, dtype=torch.float32)
        
        # Extract and map label from tactic
        tactic = data_dict.get("tactic", "unknown")
        label = self.label_to_id.get(tactic, 0)
        
        return features_tensor, label

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (feature_tensor, label)
        """
        return self.processed_data[idx]
