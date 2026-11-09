/**
 * =====================================================================
 * AIOD ENTERPRISE 3D NEURAL VISUALIZER (Three.js)
 * Isolates 3D rendering to Overview section with WebGL safety fallbacks,
 * multi-theme palette adaptation (Obsidian Cyber & Noir Gold),
 * visibility throttling, reduced-motion compliance, and resource cleanup.
 * =====================================================================
 */

(function () {
    'use strict';

    let scene, camera, renderer, animationFrameId;
    let neuralSphereGroup, particlesGeometry, particlesMesh, linesGeometry, linesMesh;
    let isInitialized = false;
    let isPaused = false;
    let currentSection = 'overview';
    let currentTheme = 'obsidian';
    let mouseX = 0, mouseY = 0;
    let targetX = 0, targetY = 0;
    let containerEl = null;

    let storedNodePositions = [];
    let nodeCount = 120;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
        } catch (e) {
            return false;
        }
    }

    function getThemeColors(theme) {
        if (theme === 'noir-gold') {
            return {
                primary: new THREE.Color(0xD4AF37),   // Gold Accent
                secondary: new THREE.Color(0xE6C65A), // Bright Gold
                tertiary: new THREE.Color(0x222222),  // Deep Obsidian Core
                lineAlphaScale: 0.55
            };
        }
        // Default Obsidian Cyber
        return {
            primary: new THREE.Color(0x3B82F6),   // Electric Blue
            secondary: new THREE.Color(0x8B5CF6), // Violet
            tertiary: new THREE.Color(0x06B6D4),  // Cyan
            lineAlphaScale: 0.4
        };
    }

    function initThreeScene(containerId) {
        containerEl = document.getElementById(containerId);
        if (!containerEl) return;

        // Detect initial theme from document attribute or storage
        currentTheme = document.documentElement.getAttribute('data-theme') || 'obsidian';

        // Graceful fallback if Three.js CDN failed or WebGL unsupported
        if (typeof window.THREE === 'undefined' || !checkWebGLSupport()) {
            console.warn('[AIOD-3D] WebGL or Three.js unavailable. Enabling ambient CSS gradient fallback.');
            containerEl.classList.add('three-fallback');
            return;
        }

        try {
            const width = window.innerWidth;
            const height = window.innerHeight;

            // Scene & Camera
            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
            camera.position.z = 240;

            // Renderer with performance optimizations
            renderer = new THREE.WebGLRenderer({
                alpha: true,
                antialias: window.devicePixelRatio <= 1.5,
                powerPreference: 'high-performance'
            });
            renderer.setSize(width, height);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.domElement.style.pointerEvents = 'none';
            renderer.domElement.style.position = 'absolute';
            renderer.domElement.style.top = '0';
            renderer.domElement.style.left = '0';
            renderer.domElement.style.width = '100%';
            renderer.domElement.style.height = '100%';

            containerEl.appendChild(renderer.domElement);

            // Construct 3D Neural Sphere & Particle Network
            buildNeuralNetwork();

            // Event Listeners
            window.addEventListener('resize', onWindowResize, { passive: true });
            document.addEventListener('pointermove', onPointerMove, { passive: true });
            document.addEventListener('visibilitychange', onVisibilityChange, { passive: true });

            isInitialized = true;

            if (prefersReducedMotion) {
                // Render single static rest frame
                renderer.render(scene, camera);
                isPaused = true;
            } else {
                startAnimationLoop();
            }

            console.info('[AIOD-3D] Three.js Neural Visualizer initialized successfully with theme:', currentTheme);

        } catch (err) {
            console.warn('[AIOD-3D] Three.js initialization caught error:', err);
            containerEl.classList.add('three-fallback');
            dispose();
        }
    }

    function buildNeuralNetwork() {
        neuralSphereGroup = new THREE.Group();
        scene.add(neuralSphereGroup);

        const isMobile = window.innerWidth < 768;
        nodeCount = isMobile ? 60 : 120;
        const radius = 70;

        storedNodePositions = [];
        const colors = [];
        const themeColors = getThemeColors(currentTheme);

        // Generate Fibonacci Sphere Nodes
        const phi = Math.PI * (3 - Math.sqrt(5));
        for (let i = 0; i < nodeCount; i++) {
            const y = 1 - (i / (nodeCount - 1)) * 2;
            const r = Math.sqrt(1 - y * y);
            const theta = phi * i;

            const x = Math.cos(theta) * r;
            const z = Math.sin(theta) * r;

            const px = x * radius + (Math.random() - 0.5) * 6;
            const py = y * radius + (Math.random() - 0.5) * 6;
            const pz = z * radius + (Math.random() - 0.5) * 6;

            storedNodePositions.push(px, py, pz);

            // Interpolate colors based on theme
            const mixColor = i % 3 === 0 ? themeColors.tertiary : (i % 2 === 0 ? themeColors.primary : themeColors.secondary);
            colors.push(mixColor.r, mixColor.g, mixColor.b);
        }

        // Particle Nodes Geometry
        particlesGeometry = new THREE.BufferGeometry();
        particlesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(storedNodePositions, 3));
        particlesGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const particlesMaterial = new THREE.PointsMaterial({
            size: isMobile ? 3.5 : 4.5,
            vertexColors: true,
            transparent: true,
            opacity: 0.88,
            blending: currentTheme === 'noir-gold' ? THREE.NormalBlending : THREE.AdditiveBlending
        });

        particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
        neuralSphereGroup.add(particlesMesh);

        // Connect nearby nodes with neural synapses (lines)
        const linePositions = [];
        const lineColors = [];
        const maxDist = 32;

        for (let i = 0; i < nodeCount; i++) {
            const x1 = storedNodePositions[i * 3];
            const y1 = storedNodePositions[i * 3 + 1];
            const z1 = storedNodePositions[i * 3 + 2];

            for (let j = i + 1; j < nodeCount; j++) {
                const x2 = storedNodePositions[j * 3];
                const y2 = storedNodePositions[j * 3 + 1];
                const z2 = storedNodePositions[j * 3 + 2];

                const dx = x1 - x2;
                const dy = y1 - y2;
                const dz = z1 - z2;
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

                if (dist < maxDist) {
                    linePositions.push(x1, y1, z1, x2, y2, z2);
                    const alpha = (1.0 - (dist / maxDist)) * themeColors.lineAlphaScale;
                    lineColors.push(themeColors.primary.r * alpha, themeColors.primary.g * alpha, themeColors.primary.b * alpha);
                    lineColors.push(themeColors.secondary.r * alpha, themeColors.secondary.g * alpha, themeColors.secondary.b * alpha);
                }
            }
        }

        linesGeometry = new THREE.BufferGeometry();
        linesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
        linesGeometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));

        const linesMaterial = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.35,
            blending: currentTheme === 'noir-gold' ? THREE.NormalBlending : THREE.AdditiveBlending
        });

        linesMesh = new THREE.LineSegments(linesGeometry, linesMaterial);
        neuralSphereGroup.add(linesMesh);

        if (!isMobile) {
            neuralSphereGroup.position.x = 45;
        }
    }

    function setTheme(newTheme) {
        currentTheme = newTheme;
        if (!isInitialized) return;

        const themeColors = getThemeColors(newTheme);

        // Update particle colors
        if (particlesGeometry && particlesGeometry.attributes.color) {
            const colorAttr = particlesGeometry.attributes.color;
            for (let i = 0; i < nodeCount; i++) {
                const mixColor = i % 3 === 0 ? themeColors.tertiary : (i % 2 === 0 ? themeColors.primary : themeColors.secondary);
                colorAttr.setXYZ(i, mixColor.r, mixColor.g, mixColor.b);
            }
            colorAttr.needsUpdate = true;
        }

        if (particlesMesh && particlesMesh.material) {
            particlesMesh.material.blending = newTheme === 'noir-gold' ? THREE.NormalBlending : THREE.AdditiveBlending;
            particlesMesh.material.needsUpdate = true;
        }

        // Update line synapse colors
        if (linesGeometry && linesGeometry.attributes.color && storedNodePositions.length > 0) {
            const colorAttr = linesGeometry.attributes.color;
            const posAttr = linesGeometry.attributes.position;
            const lineVertexCount = posAttr.count;

            for (let i = 0; i < lineVertexCount; i += 2) {
                const alpha = (i % 4 === 0) ? 0.45 : 0.25;
                colorAttr.setXYZ(i, themeColors.primary.r * alpha, themeColors.primary.g * alpha, themeColors.primary.b * alpha);
                colorAttr.setXYZ(i + 1, themeColors.secondary.r * alpha, themeColors.secondary.g * alpha, themeColors.secondary.b * alpha);
            }
            colorAttr.needsUpdate = true;
        }

        if (linesMesh && linesMesh.material) {
            linesMesh.material.blending = newTheme === 'noir-gold' ? THREE.NormalBlending : THREE.AdditiveBlending;
            linesMesh.material.opacity = newTheme === 'noir-gold' ? 0.45 : 0.28;
            linesMesh.material.needsUpdate = true;
        }

        if (renderer && scene && camera) {
            renderer.render(scene, camera);
        }
    }

    function onPointerMove(e) {
        if (!isInitialized || isPaused) return;
        mouseX = (e.clientX - window.innerWidth / 2) * 0.0006;
        mouseY = (e.clientY - window.innerHeight / 2) * 0.0006;
    }

    function onWindowResize() {
        if (!camera || !renderer) return;
        const width = window.innerWidth;
        const height = window.innerHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    }

    function onVisibilityChange() {
        if (document.hidden) {
            pause();
        } else if (currentSection === 'overview' && !prefersReducedMotion) {
            resume();
        }
    }

    function startAnimationLoop() {
        if (!isInitialized || prefersReducedMotion) return;

        function animate() {
            if (isPaused) {
                animationFrameId = null;
                return;
            }

            animationFrameId = requestAnimationFrame(animate);

            // Smooth parallax dampening
            targetX += (mouseX - targetX) * 0.05;
            targetY += (mouseY - targetY) * 0.05;

            if (neuralSphereGroup) {
                neuralSphereGroup.rotation.y += 0.0025;
                neuralSphereGroup.rotation.x += 0.0012;

                neuralSphereGroup.rotation.y += targetX * 0.4;
                neuralSphereGroup.rotation.x += targetY * 0.4;
            }

            renderer.render(scene, camera);
        }

        if (!animationFrameId) {
            animate();
        }
    }

    function setSection(sectionName) {
        currentSection = sectionName;
        if (!isInitialized) return;

        if (sectionName === 'overview') {
            if (containerEl) containerEl.style.opacity = currentTheme === 'noir-gold' ? '0.75' : '0.85';
            resume();
        } else {
            if (containerEl) containerEl.style.opacity = '0.12';
            pause();
        }
    }

    function pause() {
        isPaused = true;
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
    }

    function resume() {
        if (!isInitialized || prefersReducedMotion) return;
        if (isPaused) {
            isPaused = false;
            startAnimationLoop();
        }
    }

    function dispose() {
        pause();
        window.removeEventListener('resize', onWindowResize);
        document.removeEventListener('pointermove', onPointerMove);
        document.removeEventListener('visibilitychange', onVisibilityChange);

        if (scene) {
            scene.traverse((object) => {
                if (object.geometry) object.geometry.dispose();
                if (object.material) {
                    if (Array.isArray(object.material)) {
                        object.material.forEach((mat) => mat.dispose());
                    } else {
                        object.material.dispose();
                    }
                }
            });
        }

        if (renderer && renderer.domElement && renderer.domElement.parentNode) {
            renderer.domElement.parentNode.removeChild(renderer.domElement);
            renderer.dispose();
        }

        scene = null;
        camera = null;
        renderer = null;
        isInitialized = false;
    }

    // Expose manager safely on window
    window.ThreeSceneManager = {
        init: initThreeScene,
        setSection: setSection,
        setTheme: setTheme,
        pause: pause,
        resume: resume,
        dispose: dispose
    };

})();
