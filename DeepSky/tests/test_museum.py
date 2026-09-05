from __future__ import annotations

from pathlib import Path
import unittest

from app.web_app import app, index, museum_page, sitemap_xml


class MuseumPageTests(unittest.TestCase):
    def test_museum_route_contains_interactive_and_accessible_galleries(self) -> None:
        response = museum_page()
        html = response.body.decode("utf-8")

        self.assertIn("DeepSky Virtual Museum", html)
        self.assertIn('id="museum-canvas"', html)
        self.assertIn('src="/static/museum/museum.js"', html)
        self.assertIn('href="/static/museum/museum.css"', html)
        self.assertIn('id="museum-info"', html)
        self.assertIn('id="mobile-controls"', html)
        self.assertIn('rel="canonical" href="https://app.deepskyprocessor.com/museum"', html)
        self.assertNotIn('alt=""', html)

        for image_name in ("andromeda-m31.png", "m81.png", "veil-nebula.png"):
            self.assertIn(f"/static/museum/images/{image_name}", html)

    def test_museum_assets_and_discovery_links_exist(self) -> None:
        static_root = Path(__file__).parents[1] / "app" / "static" / "museum"
        self.assertTrue((static_root / "museum.js").is_file())
        self.assertTrue((static_root / "museum.css").is_file())
        for image_name in ("andromeda-m31.png", "m81.png", "veil-nebula.png"):
            image = static_root / "images" / image_name
            self.assertGreater(image.stat().st_size, 100_000)
            self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

        self.assertIn('href="/museum"', index())
        self.assertIn("/museum", {route.path for route in app.routes})
        self.assertIn(
            "<loc>https://app.deepskyprocessor.com/museum</loc>",
            sitemap_xml().body.decode("utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
