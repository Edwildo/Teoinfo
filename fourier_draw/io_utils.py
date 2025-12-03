"""
Utilidades para carga y guardado de contornos desde archivos e imágenes.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Tuple
from PIL import Image
from fourier_draw.contour import Contour


def load_contour_from_txt(path: Union[str, Path]) -> Contour:
    """
    Carga un contorno desde un archivo de texto.

    El archivo debe tener dos columnas separadas por espacios o comas:
    - Primera columna: coordenadas x
    - Segunda columna: coordenadas y

    Se pueden usar espacios, comas o tabs como separadores.

    Parameters
    ----------
    path : str or Path
        Ruta al archivo de texto.

    Returns
    -------
    Contour
        Contorno cargado desde el archivo.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si el archivo no tiene el formato correcto o está vacío.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"El archivo no existe: {path}")

    try:
        # Intentar cargar con numpy (soporta múltiples separadores)
        data = np.loadtxt(path, delimiter=None)

        if data.size == 0:
            raise ValueError("El archivo está vacío")

        if data.ndim != 2:
            raise ValueError(
                f"El archivo debe tener 2 columnas, pero tiene forma {data.shape}"
            )

        if data.shape[1] < 2:
            raise ValueError(
                f"El archivo debe tener al menos 2 columnas, pero tiene {data.shape[1]}"
            )

        # Extraer coordenadas x e y
        x = data[:, 0]
        y = data[:, 1]

        return Contour(x, y)

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error al leer el archivo {path}: {e}")


def save_contour_to_txt(contour: Contour, path: Union[str, Path]) -> None:
    """
    Guarda un contorno en un archivo de texto.

    El archivo se guarda con dos columnas separadas por espacios:
    - Primera columna: coordenadas x
    - Segunda columna: coordenadas y

    Parameters
    ----------
    contour : Contour
        Contorno a guardar.
    path : str or Path
        Ruta donde guardar el archivo.

    Raises
    ------
    ValueError
        Si el contorno está vacío.
    """
    path = Path(path)

    if contour.n_points == 0:
        raise ValueError("No se puede guardar un contorno vacío")

    # Crear array con x e y
    data = np.column_stack([contour.x, contour.y])

    # Guardar con formato legible
    np.savetxt(path, data, fmt="%.10f", delimiter=" ")


def load_contour_from_image(
    path: Union[str, Path],
    threshold: int = 128,
    method: str = "largest",
    invert: bool = False,
) -> Contour:
    """
    Carga un contorno desde una imagen.

    Extrae el contorno de una imagen convirtiéndola a escala de grises,
    aplicando un umbral y detectando los bordes.

    Parameters
    ----------
    path : str or Path
        Ruta a la imagen (PNG, JPG, etc.).
    threshold : int
        Umbral para binarización (0-255). Valores por debajo se consideran
        fondo, valores por encima objeto.
    method : str
        Método para seleccionar contorno:
        - "largest": Usa el contorno más grande (default)
        - "outer": Usa el contorno externo
        - "all": Retorna el primer contorno encontrado
    invert : bool
        Si True, invierte la imagen antes de procesar (útil para imágenes
        con fondo oscuro).

    Returns
    -------
    Contour
        Contorno extraído de la imagen.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si no se puede extraer un contorno de la imagen.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"El archivo no existe: {path}")

    try:
        # Cargar imagen
        img = Image.open(path)

        # Convertir a escala de grises si es necesario
        if img.mode != "L":
            img = img.convert("L")

        # Convertir a array numpy
        img_array = np.array(img, dtype=np.uint8)

        # Invertir si es necesario
        if invert:
            img_array = 255 - img_array

        # Aplicar umbral para binarizar
        binary = img_array > threshold

        # Extraer contorno usando marching squares o método simple
        contour_points = _extract_contour_from_binary(binary, method)

        if len(contour_points) == 0:
            raise ValueError(
                f"No se pudo extraer un contorno de la imagen. "
                f"Intenta ajustar el umbral (threshold={threshold}) o invertir (invert={invert})"
            )

        # Convertir a Contour
        x = contour_points[:, 0]
        y = contour_points[:, 1]

        return Contour(x, y)

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error al procesar la imagen {path}: {e}")


def _extract_contour_from_binary(
    binary: np.ndarray, method: str = "largest"
) -> np.ndarray:
    """
    Extrae contornos de una imagen binaria.

    Parameters
    ----------
    binary : np.ndarray
        Imagen binaria (True = objeto, False = fondo).
    method : str
        Método para seleccionar contorno.

    Returns
    -------
    np.ndarray
        Array de forma (N, 2) con coordenadas [x, y] del contorno.
    """
    # Encontrar todos los píxeles de borde
    # Un píxel es borde si es True y tiene al menos un vecino False
    h, w = binary.shape
    edges = []

    # Optimización: usar operaciones vectorizadas de numpy para detectar bordes
    # Método más eficiente: usar convolución 2D
    from scipy import ndimage
    
    try:
        # Detectar bordes usando operación morfológica
        # Un píxel es borde si es True y tiene al menos un vecino False
        # Usar scipy.ndimage para detectar bordes eficientemente
        structure = np.ones((3, 3), dtype=bool)
        dilated = ndimage.binary_dilation(binary, structure=structure)
        edges_mask = dilated & ~binary  # Bordes externos
        edges_mask = edges_mask | (binary & ~ndimage.binary_erosion(binary, structure=structure))  # Bordes internos
        
        # Obtener coordenadas de los bordes
        y_coords, x_coords = np.where(edges_mask)
        edges = np.column_stack([x_coords, y_coords]).astype(np.float64)
        
    except ImportError:
        # Fallback si scipy no está disponible: método optimizado con numpy
        # Crear una imagen con padding
        padded = np.pad(binary, 1, mode='constant', constant_values=False)
        
        # Detectar bordes usando slicing vectorizado
        edges = []
        for y in range(1, h + 1):
            row_edges = []
            for x in range(1, w + 1):
                if padded[y, x]:
                    # Verificar vecinos con slicing
                    neighbors = padded[y-1:y+2, x-1:x+2]
                    if not np.all(neighbors):
                        row_edges.append([x-1, y-1])
            if row_edges:
                edges.extend(row_edges)
        
        edges = np.array(edges, dtype=np.float64) if edges else np.array([], dtype=np.float64).reshape(0, 2)

    if len(edges) == 0:
        return np.array([], dtype=np.float64).reshape(0, 2)

    edges = np.array(edges, dtype=np.float64)

    # Ordenar puntos para formar un contorno continuo
    if method == "largest" or method == "outer":
        # Encontrar el punto más a la izquierda (o más arriba si hay empate)
        start_idx = np.lexsort((edges[:, 1], edges[:, 0]))[0]
        ordered = _order_contour_points(edges, start_idx)
    else:
        # Retornar todos los puntos sin ordenar
        ordered = edges

    return ordered


def _order_contour_points(points: np.ndarray, start_idx: int) -> np.ndarray:
    """
    Ordena puntos de contorno para formar una secuencia continua.

    Usa un algoritmo de vecino más cercano para conectar los puntos.

    Parameters
    ----------
    points : np.ndarray
        Array de forma (N, 2) con coordenadas [x, y].
    start_idx : int
        Índice del punto inicial.

    Returns
    -------
    np.ndarray
        Puntos ordenados de forma (M, 2).
    """
    if len(points) == 0:
        return points

    if len(points) == 1:
        return points

    ordered = [points[start_idx].copy()]
    remaining = set(range(len(points)))
    remaining.remove(start_idx)
    current = points[start_idx]

    while remaining:
        # Encontrar el punto más cercano
        min_dist = float("inf")
        next_idx = None

        for idx in remaining:
            dist = np.sqrt(
                (points[idx, 0] - current[0]) ** 2 + (points[idx, 1] - current[1]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                next_idx = idx

        if next_idx is None:
            break

        ordered.append(points[next_idx].copy())
        remaining.remove(next_idx)
        current = points[next_idx]

    return np.array(ordered, dtype=np.float64)


def load_contour_auto(path: Union[str, Path], **kwargs) -> Contour:
    """
    Carga un contorno automáticamente detectando si es imagen o archivo de texto.

    Parameters
    ----------
    path : str or Path
        Ruta al archivo (imagen o texto).
    **kwargs
        Argumentos adicionales pasados a load_contour_from_image si es una imagen.

    Returns
    -------
    Contour
        Contorno cargado.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si el formato no es soportado.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"El archivo no existe: {path}")

    # Detectar tipo de archivo por extensión
    ext = path.suffix.lower()

    # Extensiones de imagen comunes
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif"}

    if ext in image_extensions:
        return load_contour_from_image(path, **kwargs)
    else:
        # Asumir que es un archivo de texto
        return load_contour_from_txt(path)


