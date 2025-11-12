import torch
import numpy as np
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import shapely
import json
import os
import hashlib

class BorderDistanceMapper:
    """
    A ComfyUI node that draws a buffer outline around country borders on a world map
    """
    
    # Cache file paths
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    COUNTRIES_CACHE = os.path.join(CACHE_DIR, "world_countries_cache.json")
    WORLD_MAP_CACHE = os.path.join(CACHE_DIR, "world_maps")
    
    def __init__(self):
        # Create cache directory if it doesn't exist
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.WORLD_MAP_CACHE, exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "city_name": ("STRING", {
                    "default": "London",
                    "multiline": False
                }),
                "distance_km": ("FLOAT", {
                    "default": 20000.0,
                    "min": 1.0,
                    "max": 40000.0,
                    "step": 100.0
                }),
                "image_width": ("INT", {
                    "default": 2048,
                    "min": 1024,
                    "max": 4096,
                    "step": 64
                }),
                "image_height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 4096,
                    "step": 64
                }),
                "outline_color": (["red", "blue", "green", "yellow", "white", "black", "orange"], {
                    "default": "red"
                }),
                "show_country": ("BOOLEAN", {
                    "default": True
                }),
                "map_style": (["detailed", "simple", "minimal", "coastline"], {
                    "default": "detailed"
                }),
                "detail_level": (["low", "medium", "high", "ultra"], {
                    "default": "high"
                }),
                "force_refresh": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("world_map", "buffer_mask")
    FUNCTION = "draw_border_buffer"
    CATEGORY = "image/mapping"
    
    def get_world_map_cache_key(self, image_width, image_height, map_style, detail_level):
        """Generate a unique cache key for world map settings"""
        key_string = f"{image_width}x{image_height}_{map_style}_{detail_level}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_world_map(self, image_width, image_height, map_style, detail_level):
        """Get cached world map if it exists"""
        cache_key = self.get_world_map_cache_key(image_width, image_height, map_style, detail_level)
        cache_file = os.path.join(self.WORLD_MAP_CACHE, f"{cache_key}.png")
        
        if os.path.exists(cache_file):
            try:
                print(f"Loading cached world map: {cache_file}")
                img = Image.open(cache_file).convert('RGB')
                # Verify dimensions match
                if img.size == (image_width, image_height):
                    draw = ImageDraw.Draw(img)
                    return img, draw
                else:
                    print("Cached map dimensions don't match, regenerating...")
            except Exception as e:
                print(f"Error loading cached world map: {e}")
        
        return None, None
    
    def save_world_map_to_cache(self, image, image_width, image_height, map_style, detail_level):
        """Save world map to cache for future use"""
        try:
            cache_key = self.get_world_map_cache_key(image_width, image_height, map_style, detail_level)
            cache_file = os.path.join(self.WORLD_MAP_CACHE, f"{cache_key}.png")
            
            # Save as PNG to preserve quality
            image.save(cache_file, "PNG")
            print(f"World map saved to cache: {cache_file}")
        except Exception as e:
            print(f"Warning: Could not save world map to cache: {e}")
    
    def get_country_from_city(self, city_name):
        """Get country code from city name using Nominatim API"""
        try:
            url = f"https://nominatim.openstreetmap.org/search"
            params = {
                'q': city_name,
                'format': 'json',
                'limit': 1
            }
            headers = {'User-Agent': 'ComfyUI-BorderDistance/1.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            
            if data:
                # Get detailed info
                place_id = data[0]['place_id']
                detail_url = f"https://nominatim.openstreetmap.org/details.php"
                detail_params = {
                    'place_id': place_id,
                    'format': 'json'
                }
                detail_response = requests.get(detail_url, params=detail_params, headers=headers, timeout=10)
                detail_data = detail_response.json()
                
                country_code = detail_data.get('country_code', '').upper()
                return country_code
        except Exception as e:
            print(f"Error getting country: {e}")
            return None
    
    def get_country_border_points(self, country_code):
        """Get detailed border points for a country using Overpass API"""
        try:
            overpass_url = "http://overpass-api.de/api/interpreter"
            
            # Query for detailed country boundaries
            query = f"""
            [out:json][timeout:90];
            relation["boundary"="administrative"]["admin_level"="2"]["ISO3166-1"="{country_code}"];
            out geom;
            """
            
            print(f"Fetching borders for {country_code}...")
            response = requests.post(overpass_url, data=query, timeout=90)
            
            if response.status_code != 200:
                print(f"Overpass API error: {response.status_code}")
                return []
                
            data = response.json()
            
            border_points = []
            if 'elements' in data:
                for element in data['elements']:
                    if element['type'] == 'relation':
                        # Process relation members with geometry
                        if 'members' in element:
                            for member in element['members']:
                                if 'geometry' in member and member.get('type') == 'node':
                                    for point in member['geometry']:
                                        border_points.append((point['lon'], point['lat']))
            
            # If no points found, try alternative query
            if not border_points:
                return self.get_country_border_points_alternative(country_code)
                
            print(f"Found {len(border_points)} border points for {country_code}")
            return border_points
            
        except Exception as e:
            print(f"Error getting border points: {e}")
            return []
    
    def get_country_border_points_alternative(self, country_code):
        """Alternative method to get country borders"""
        try:
            overpass_url = "http://overpass-api.de/api/interpreter"
            
            query = f"""
            [out:json][timeout:90];
            area["ISO3166-1"="{country_code}"][admin_level=2]->.searchArea;
            (
              relation(area.searchArea)["boundary"="administrative"]["admin_level"="2"];
            );
            out geom;
            """
            
            response = requests.post(overpass_url, data=query, timeout=90)
            data = response.json()
            
            border_points = []
            if 'elements' in data:
                for element in data['elements']:
                    if 'geometry' in element:
                        for point in element['geometry']:
                            border_points.append((point['lon'], point['lat']))
                    elif 'members' in element:
                        for member in element['members']:
                            if 'geometry' in member:
                                for point in member['geometry']:
                                    border_points.append((point['lon'], point['lat']))
            
            return border_points
            
        except Exception as e:
            print(f"Error in alternative border fetch: {e}")
            return []
    
    def lat_lon_to_world_pixels(self, lat, lon, width, height):
        """Convert lat/lon to pixel coordinates on world map (Mercator projection)"""
        try:
            # Mercator projection
            x = (lon + 180) * (width / 360)
            
            # Mercator y coordinate with bounds checking
            lat_bound = max(min(lat, 85.0), -85.0)  # Prevent infinity at poles
            lat_rad = np.radians(lat_bound)
            merc_y = np.log(np.tan((np.pi / 4) + (lat_rad / 2)))
            y = (height / 2) - (height * merc_y / (2 * np.pi))
            
            return int(x), int(y)
        except Exception as e:
            print(f"Error in coordinate conversion: {e}")
            return 0, 0
    
    def remove_duplicate_points(self, points):
        """Remove duplicate points using a method that works with coordinate tuples"""
        unique_points = []
        seen = set()
        
        for point in points:
            # Convert to tuple of rounded coordinates to handle floating point precision
            rounded_point = (round(point[0], 6), round(point[1], 6))
            if rounded_point not in seen:
                seen.add(rounded_point)
                unique_points.append(point)
        
        return unique_points
    
    def create_buffer_polygon(self, border_points, distance_km):
        """Create a buffer polygon around border points"""
        if not border_points or len(border_points) < 3:
            print("Not enough border points for polygon")
            return None
        
        try:
            # Remove duplicate points using our custom function
            unique_points = self.remove_duplicate_points(border_points)
            
            if len(unique_points) < 3:
                print("Not enough unique points for polygon")
                return None
            
            print(f"Creating polygon with {len(unique_points)} points...")
            
            # Create polygon from border points
            polygon = Polygon(unique_points)
            
            if not polygon.is_valid:
                print("Invalid polygon, trying to fix with buffer(0)...")
                polygon = polygon.buffer(0)  # Fix self-intersections
            
            if not polygon.is_valid:
                print("Polygon still invalid, trying convex hull as fallback...")
                # Use convex hull as last resort
                if len(unique_points) >= 3:
                    points_array = np.array(unique_points)
                    hull = ConvexHull(points_array)
                    hull_points = [points_array[i] for i in hull.vertices]
                    polygon = Polygon(hull_points)
                else:
                    return None
            
            # Buffer distance in degrees (approximate)
            buffer_deg = distance_km / 111.0
            
            print(f"Creating buffer with {buffer_deg:.6f} degree radius...")
            
            # Create buffer with higher resolution for better detail
            buffered = polygon.buffer(buffer_deg, resolution=16)
            
            if buffered.is_empty:
                print("Buffer created empty polygon")
                return None
                
            print("Buffer created successfully")
            return buffered
            
        except Exception as e:
            print(f"Error creating buffer: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_all_countries(self):
        """Get all country borders from cache or Overpass API"""
        # Check if cache file exists
        if os.path.exists(self.COUNTRIES_CACHE):
            try:
                print(f"Loading countries from cache: {self.COUNTRIES_CACHE}")
                with open(self.COUNTRIES_CACHE, 'r') as f:
                    countries = json.load(f)
                print(f"Loaded {len(countries)} countries from cache")
                return countries
            except Exception as e:
                print(f"Error loading cache: {e}, fetching fresh data...")
        
        # Fetch from API if cache doesn't exist
        try:
            overpass_url = "http://overpass-api.de/api/interpreter"
            # High detail query for country boundaries
            query = """
            [out:json][timeout:300];
            relation["boundary"="administrative"]["admin_level"="2"]["ISO3166-1"];
            out geom;
            """
            
            print("Fetching detailed country borders from OpenStreetMap...")
            response = requests.post(overpass_url, data=query, timeout=300)
            data = response.json()
            
            countries = {}
            if 'elements' in data:
                for element in data['elements']:
                    country_code = element.get('tags', {}).get('ISO3166-1', '')
                    if country_code and 'members' in element:
                        border_points = []
                        for member in element['members']:
                            if 'geometry' in member:
                                for point in member['geometry']:
                                    border_points.append((point['lon'], point['lat']))
                        if border_points and len(border_points) >= 10:
                            countries[country_code] = border_points
            
            print(f"Loaded {len(countries)} detailed countries")
            
            # Save to cache
            try:
                print(f"Saving countries to cache: {self.COUNTRIES_CACHE}")
                with open(self.COUNTRIES_CACHE, 'w') as f:
                    json.dump(countries, f)
                print("Cache saved successfully")
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
            
            return countries
            
        except Exception as e:
            print(f"Error getting all countries: {e}")
            return {}
    
    def draw_detailed_country_outlines(self, draw, all_countries, width, height, detail_level="high"):
        """Draw detailed country outlines"""
        print(f"Drawing {detail_level} detail country outlines...")
        
        # Adjust parameters based on detail level
        detail_config = {
            "low": {"line_width": 1, "max_point_distance": 200},
            "medium": {"line_width": 1, "max_point_distance": 100},
            "high": {"line_width": 1, "max_point_distance": 50},
            "ultra": {"line_width": 1, "max_point_distance": 25}
        }
        
        config = detail_config[detail_level]
        outline_color = (60, 60, 60)  # Dark gray for country borders
        
        countries_drawn = 0
        
        for country_code, border_points in all_countries.items():
            if len(border_points) < 10:
                continue
                
            try:
                # Convert border points to pixels
                country_pixels = []
                for lon, lat in border_points:
                    x, y = self.lat_lon_to_world_pixels(lat, lon, width, height)
                    country_pixels.append((x, y))
                
                # Draw the outline
                if len(country_pixels) > 1:
                    for i in range(len(country_pixels) - 1):
                        x1, y1 = country_pixels[i]
                        x2, y2 = country_pixels[i+1]
                        distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
                        if distance < config["max_point_distance"]:
                            draw.line([country_pixels[i], country_pixels[i+1]], 
                                     fill=outline_color, width=config["line_width"])
                    
                    countries_drawn += 1
                    
            except Exception as e:
                print(f"Error drawing {country_code}: {e}")
                continue
        
        print(f"Successfully drew {countries_drawn} country outlines at {detail_level} detail")
    
    def generate_world_map_base(self, width, height, all_countries=None, map_style="detailed", detail_level="high"):
        """Generate a new world map base (only called when cache doesn't exist)"""
        print(f"Generating new world map base: {width}x{height}, {map_style}, {detail_level}")
        
        # Try to load high-quality world map images first
        map_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/8/83/Equirectangular_projection_SW.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/2/23/Blue_Marble_2002.png",
        ]
        
        for map_url in map_urls:
            try:
                print(f"Loading world map image from {map_url}...")
                response = requests.get(map_url, timeout=30)
                if response.status_code == 200:
                    map_img = Image.open(BytesIO(response.content)).convert('RGB')
                    # Use high-quality resampling
                    map_img = map_img.resize((width, height), Image.Resampling.LANCZOS)
                    draw = ImageDraw.Draw(map_img)
                    
                    # Overlay country borders if available
                    if all_countries and map_style in ["detailed", "coastline"]:
                        self.draw_detailed_country_outlines(draw, all_countries, width, height, detail_level)
                    
                    print("World map base generated successfully")
                    return map_img, draw
            except Exception as e:
                print(f"Could not load map from {map_url}: {e}")
                continue
        
        # Fallback: Create base map from scratch
        print("Creating base map from geographic data...")
        
        if map_style == "minimal":
            img = Image.new('RGB', (width, height), color=(255, 255, 255))
        elif map_style == "coastline":
            img = Image.new('RGB', (width, height), color=(100, 150, 200))
        else:
            img = Image.new('RGB', (width, height), color=(200, 220, 240))
        
        draw = ImageDraw.Draw(img)
        
        # Draw country outlines
        if all_countries:
            self.draw_detailed_country_outlines(draw, all_countries, width, height, detail_level)
        
        # Add grid for detailed modes
        if map_style in ["detailed", "coastline"]:
            grid_color = (180, 180, 180)
            for lat in range(-80, 90, 10):
                points = []
                for lon in range(-180, 181, 5):
                    x, y = self.lat_lon_to_world_pixels(lat, lon, width, height)
                    points.append((x, y))
                if len(points) > 1:
                    draw.line(points, fill=grid_color, width=1)
        
        return img, draw
    
    def get_world_map_base(self, image_width, image_height, map_style, detail_level, force_refresh=False):
        """Get world map base - uses cache unless force_refresh is True"""
        # Try to get from cache first
        if not force_refresh:
            cached_result = self.get_cached_world_map(image_width, image_height, map_style, detail_level)
            if cached_result[0] is not None:
                return cached_result
        
        # Generate new world map
        all_countries = self.get_all_countries()
        img, draw = self.generate_world_map_base(image_width, image_height, all_countries, map_style, detail_level)
        
        # Save to cache for future use
        self.save_world_map_to_cache(img, image_width, image_height, map_style, detail_level)
        
        return img, draw
    
    def create_buffer_mask(self, border_points, buffered_polygon, image_width, image_height, outline_color, show_country):
        """Create a transparent image with just the buffer and country outline"""
        # Create transparent image
        img = Image.new('RGBA', (image_width, image_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Color mapping for RGBA
        colors = {
            "red": (255, 0, 0, 255),
            "blue": (0, 0, 255, 255),
            "green": (0, 200, 0, 255),
            "yellow": (255, 255, 0, 255),
            "white": (255, 255, 255, 255),
            "black": (0, 0, 0, 255),
            "orange": (255, 140, 0, 255)
        }
        
        color = colors[outline_color]
        country_color = (255, 0, 0, 255)
        
        if buffered_polygon:
            # Draw target country outline if requested
            if show_country and border_points:
                country_pixels = []
                for lon, lat in border_points:
                    x, y = self.lat_lon_to_world_pixels(lat, lon, image_width, image_height)
                    country_pixels.append((x, y))
                
                if len(country_pixels) > 1:
                    for i in range(len(country_pixels) - 1):
                        x1, y1 = country_pixels[i]
                        x2, y2 = country_pixels[i+1]
                        distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
                        if distance < 50:
                            draw.line([country_pixels[i], country_pixels[i+1]], 
                                     fill=country_color, width=3)
            
            # Draw buffer outline
            if isinstance(buffered_polygon, MultiPolygon):
                polygons = list(buffered_polygon.geoms)
            else:
                polygons = [buffered_polygon]
            
            for poly in polygons:
                try:
                    exterior_coords = list(poly.exterior.coords)
                    buffer_pixels = []
                    for lon, lat in exterior_coords:
                        x, y = self.lat_lon_to_world_pixels(lat, lon, image_width, image_height)
                        buffer_pixels.append((x, y))
                    
                    if len(buffer_pixels) > 2:
                        for i in range(len(buffer_pixels) - 1):
                            draw.line([buffer_pixels[i], buffer_pixels[i+1]], 
                                     fill=color, width=4)
                        draw.line([buffer_pixels[-1], buffer_pixels[0]], 
                                 fill=color, width=4)
                            
                except Exception as e:
                    print(f"Error drawing buffer in mask: {e}")
                    continue
        else:
            # Fallback circle
            try:
                lons = [p[0] for p in border_points]
                lats = [p[1] for p in border_points]
                center_lon = sum(lons) / len(lons)
                center_lat = sum(lats) / len(lats)
                
                center_x, center_y = self.lat_lon_to_world_pixels(center_lat, center_lon, image_width, image_height)
                radius_pixels = int((distance_km / 111.0) * (image_width / 360.0))
                
                bbox = [
                    center_x - radius_pixels, center_y - radius_pixels,
                    center_x + radius_pixels, center_y + radius_pixels
                ]
                draw.ellipse(bbox, outline=color, width=4)
                
            except Exception as e:
                print(f"Error drawing fallback circle in mask: {e}")
        
        return img
    
    def draw_border_buffer(self, city_name, distance_km, image_width, image_height, 
                           outline_color, show_country, map_style, detail_level, force_refresh):
        """Main function - uses cached world maps for performance"""
        
        print("=== Border Distance Mapper ===")
        print(f"Settings: {image_width}x{image_height}, {map_style}, {detail_level}")
        print(f"City: {city_name}, Distance: {distance_km}km")
        
        # Color mapping
        colors = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 200, 0),
            "yellow": (255, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "orange": (255, 140, 0)
        }
        
        # Get country from city
        print(f"Looking up country for city: {city_name}")
        country_code = self.get_country_from_city(city_name)
        
        if not country_code:
            print(f"Could not find country for city: {city_name}")
            fallback_countries = {
                "london": "GB", "paris": "FR", "berlin": "DE", "tokyo": "JP",
                "new york": "US", "beijing": "CN", "moscow": "RU", "rome": "IT",
                "madrid": "ES", "amsterdam": "NL", "brussels": "BE", "vienna": "AT",
                "prague": "CZ", "budapest": "HU", "warsaw": "PL", "stockholm": "SE"
            }
            country_code = fallback_countries.get(city_name.lower())
            
            if not country_code:
                return self.create_error_outputs(image_width, image_height, f"Could not find country for: {city_name}")
            else:
                print(f"Using fallback country: {country_code}")
        
        print(f"Found country: {country_code}")
        
        # Get world map base (uses cache unless force_refresh=True)
        print("Loading world map base...")
        world_map_img, world_map_draw = self.get_world_map_base(image_width, image_height, map_style, detail_level, force_refresh)
        
        # Get country border points
        all_countries = self.get_all_countries()
        border_points = all_countries.get(country_code, [])
        
        if not border_points:
            print("Border points not in cache, fetching from API...")
            border_points = self.get_country_border_points(country_code)
        
        if not border_points:
            print(f"Could not fetch border points for {country_code}")
            world_map_draw.text((50, 50), f"No border data for: {country_code}", fill=(255, 0, 0))
            world_map_draw.text((50, 80), f"City: {city_name}", fill=(255, 0, 0))
            img_array = np.array(world_map_img).astype(np.float32) / 255.0
            
            mask_error = Image.new('RGBA', (image_width, image_height), (0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_error)
            mask_draw.text((50, 50), f"No data: {country_code}", fill=(255, 0, 0, 255))
            mask_array = np.array(mask_error).astype(np.float32) / 255.0
            
            return (torch.from_numpy(img_array)[None,], torch.from_numpy(mask_array)[None,])
        
        print(f"Found {len(border_points)} border points for {country_code}")
        
        # Create buffer polygon
        print("Creating buffer zone...")
        buffered_polygon = self.create_buffer_polygon(border_points, distance_km)
        
        # Create buffer mask
        print("Creating buffer mask...")
        buffer_mask = self.create_buffer_mask(border_points, buffered_polygon, image_width, image_height, outline_color, show_country)
        
        # Draw buffer and country on world map
        print("Drawing overlay on world map...")
        self.draw_overlay_on_world_map(world_map_draw, border_points, buffered_polygon, colors[outline_color], show_country, distance_km)
        
        # Add title
        title_color = (0, 0, 0) if map_style != "minimal" else (100, 100, 100)
        world_map_draw.text((20, 20), f"{city_name.title()} ({country_code})", fill=title_color)
        world_map_draw.text((20, 45), f"{distance_km}km buffer zone", fill=title_color)
        
        # Convert to tensors
        img_array = np.array(world_map_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        
        mask_array = np.array(buffer_mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array)[None,]
        
        print("=== Generation Complete ===")
        return (img_tensor, mask_tensor)
    
    def draw_overlay_on_world_map(self, draw, border_points, buffered_polygon, color, show_country, distance_km):
        """Draw buffer and country overlay on existing world map"""
        width, height = draw.im.size
        
        if buffered_polygon:
            # Draw target country outline
            if show_country and border_points:
                country_pixels = []
                for lon, lat in border_points:
                    x, y = self.lat_lon_to_world_pixels(lat, lon, width, height)
                    country_pixels.append((x, y))
                
                if len(country_pixels) > 1:
                    for i in range(len(country_pixels) - 1):
                        x1, y1 = country_pixels[i]
                        x2, y2 = country_pixels[i+1]
                        distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
                        if distance < 50:
                            draw.line([country_pixels[i], country_pixels[i+1]], 
                                     fill=(255, 0, 0), width=3)
            
            # Draw buffer outline
            if isinstance(buffered_polygon, MultiPolygon):
                polygons = list(buffered_polygon.geoms)
            else:
                polygons = [buffered_polygon]
            
            for poly in polygons:
                try:
                    exterior_coords = list(poly.exterior.coords)
                    buffer_pixels = []
                    for lon, lat in exterior_coords:
                        x, y = self.lat_lon_to_world_pixels(lat, lon, width, height)
                        buffer_pixels.append((x, y))
                    
                    if len(buffer_pixels) > 2:
                        for i in range(len(buffer_pixels) - 1):
                            draw.line([buffer_pixels[i], buffer_pixels[i+1]], 
                                     fill=color, width=4)
                        draw.line([buffer_pixels[-1], buffer_pixels[0]], 
                                 fill=color, width=4)
                    
                except Exception as e:
                    print(f"Error drawing buffer overlay: {e}")
    
    def create_error_outputs(self, image_width, image_height, error_message):
        """Create error outputs for both world map and buffer mask"""
        img_error = Image.new('RGB', (image_width, image_height), color=(50, 50, 50))
        draw_error = ImageDraw.Draw(img_error)
        draw_error.text((50, 50), error_message, fill=(255, 255, 255))
        img_array = np.array(img_error).astype(np.float32) / 255.0
        
        mask_error = Image.new('RGBA', (image_width, image_height), (0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_error)
        mask_draw.text((50, 50), error_message, fill=(255, 0, 0, 255))
        mask_array = np.array(mask_error).astype(np.float32) / 255.0
        
        return (torch.from_numpy(img_array)[None,], torch.from_numpy(mask_array)[None,])


NODE_CLASS_MAPPINGS = {
    "BorderDistanceMapper": BorderDistanceMapper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BorderDistanceMapper": "Border Distance Mapper"
}