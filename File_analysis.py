import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import h5py
import numpy as np
import logging
from io import BytesIO
from PIL import Image
import os
import platform
import tkinter as tk
from tkinter import filedialog
import rasterio
from rasterio.plot import show

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SatelliteDataProcessor:
    def __init__(self):
        self.data = None
        self.metadata = {}
        self.supported_formats = ['csv', 'json', 'h5', 'hdf5', 'jpg', 'jpeg', 'png', 'tif', 'tiff']

    def detect_file_type(self, file_path):
        """Detect the file type based on extension."""
        try:
            ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            if ext == 'zip':
                return 'zip'
            elif ext in self.supported_formats:
                return ext
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
            logger.error(f"Error detecting file type: {e}")
            raise

    def extract_zip(self, file_content):
        """Extract files from a ZIP archive in memory."""
        try:
            with zipfile.ZipFile(BytesIO(file_content), 'r') as zip_ref:
                file_list = zip_ref.namelist()
                for file_name in file_list:
                    ext = os.path.splitext(file_name)[1].lower().lstrip('.')
                    if ext in self.supported_formats:
                        with zip_ref.open(file_name) as file:
                            return file_name, file.read()
                logger.error(f"No supported files found in ZIP archive. Contents: {file_list}")
                raise ValueError(f"No supported files found in ZIP archive. Contents: {file_list}")
        except Exception as e:
            logger.error(f"Error extracting ZIP: {e}")
            raise

    def load_data(self, file_path, file_content):
        """Load data from various file types."""
        try:
            file_type = self.detect_file_type(file_path)
            logger.info(f"Processing file: {file_path} (Type: {file_type})")

            if file_type == 'zip':
                extracted_file_name, extracted_content = self.extract_zip(file_content)
                file_type = self.detect_file_type(extracted_file_name)
                file_path = extracted_file_name
            else:
                extracted_content = file_content

            if file_type == 'csv':
                self.data = pd.read_csv(BytesIO(extracted_content))
            elif file_type == 'json':
                self.data = pd.read_json(BytesIO(extracted_content))
            elif file_type in ['h5', 'hdf5']:
                with h5py.File(BytesIO(extracted_content), 'r') as h5file:
                    dataset_name = list(h5file.keys())[0]  # Assume first dataset
                    self.data = pd.DataFrame(h5file[dataset_name][:])
                    self.metadata = dict(h5file.attrs)
            elif file_type in ['jpg', 'jpeg', 'png']:
                self.data = Image.open(BytesIO(extracted_content))
                self.metadata = {'format': file_type, 'size': self.data.size}
            elif file_type in ['tif', 'tiff']:
                with rasterio.open(BytesIO(extracted_content)) as dataset:
                    self.data = dataset.read()  # Read as NumPy array
                    self.metadata = dict(dataset.profile)
                    self.metadata['raster'] = True
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            logger.info(f"Data loaded successfully from {file_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def process_data(self):
        """Process loaded data for analysis."""
        try:
            if isinstance(self.data, pd.DataFrame):
                # Handle missing values
                self.data.fillna(self.data.mean(numeric_only=True), inplace=True)
                # Convert datetime columns if present
                for col in self.data.columns:
                    if self.data[col].dtype == 'object':
                        try:
                            self.data[col] = pd.to_datetime(self.data[col])
                        except:
                            pass
                logger.info("Data processed: Missing values handled, datetime converted")
            elif isinstance(self.data, Image.Image):
                logger.info("Image data ready for visualization")
            elif isinstance(self.data, np.ndarray) and self.metadata.get('raster'):
                logger.info("Raster data ready for visualization")
            else:
                raise ValueError("Unsupported data type for processing")
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

    def generate_table(self):
        """Generate and return data in tabular format."""
        try:
            if isinstance(self.data, pd.DataFrame):
                return self.data.to_string(index=False)
            elif isinstance(self.data, Image.Image):
                return f"Image Data: {self.metadata}"
            elif isinstance(self.data, np.ndarray) and self.metadata.get('raster'):
                # For raster data, show metadata and sample values
                sample_data = pd.DataFrame(self.data[0].flatten()[:10], columns=['Sample Values'])
                return f"Raster Data (Metadata: {self.metadata})\nSample Values:\n{sample_data.to_string(index=False)}"
            else:
                raise ValueError("No tabular data available")
        except Exception as e:
            logger.error(f"Error generating table: {e}")
            raise

    def visualize_data(self):
        """Generate visualizations based on data type."""
        try:
            plt.figure(figsize=(10, 6))
            if isinstance(self.data, pd.DataFrame):
                # Plot numeric columns
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    for col in numeric_cols[:2]:  # Limit to first two for simplicity
                        plt.plot(self.data.index, self.data[col], label=col)
                    plt.title("Satellite Data Visualization")
                    plt.xlabel("Index")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(True)
                else:
                    raise ValueError("No numeric columns for plotting")
            elif isinstance(self.data, Image.Image):
                plt.imshow(self.data)
                plt.title("Satellite Image")
                plt.axis('off')
            elif isinstance(self.data, np.ndarray) and self.metadata.get('raster'):
                show(self.data, title="Satellite Raster Data")
            else:
                raise ValueError("Unsupported data type for visualization")
            plt.show()
            logger.info("Visualization generated")
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            raise


def get_file_input():
    """Open a file dialog to select a file and read its content."""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        file_path = filedialog.askopenfilename(
            title="Select a satellite data file",
            filetypes=[
                ("All supported files", "*.csv;*.json;*.h5;*.hdf5;*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.zip"),
                ("ZIP files", "*.zip"),
                ("CSV files", "*.csv"),
                ("JSON files", "*.json"),
                ("HDF5 files", "*.h5;*.hdf5"),
                ("Image files", "*.jpg;*.jpeg;*.png"),
                ("GeoTIFF files", "*.tif;*.tiff")
            ]
        )
        root.destroy()  # Close the tkinter instance

        if not file_path:
            raise ValueError("No file selected")

        logger.info(f"Selected file: {file_path}")
        with open(file_path, 'rb') as f:
            file_content = f.read()
        return file_path, file_content
    except Exception as e:
        logger.error(f"Error selecting file: {e}")
        raise


def main():
    processor = SatelliteDataProcessor()

    # Get file input from user
    try:
        file_path, file_content = get_file_input()
        processor.load_data(file_path, file_content)
        processor.process_data()

        # Generate and print table
        print("Tabular Data:")
        print(processor.generate_table())

        # Generate visualization
        processor.visualize_data()
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")


if platform.system() == "Emscripten":
    import asyncio

    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        main()