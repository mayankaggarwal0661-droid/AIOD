/**
 * =====================================================================
 * AIOD ENTERPRISE 3D NEURAL VISUALIZER (Three.js)
 * Isolates 3D rendering to Overview section with WebGL safety fallbacks,
 * visibility throttling, reduced-motion compliance, and resource cleanup.
 * =====================================================================
 */

(function () {
    'use strict';

    let scene, camera, renderer, animationFrameId;
    let neuralSphereGroup, particlesGeometry, particlesMesh, linesMesh;
    let isInitialized = false;
    let isPaused = false;
    let currentSection = 'overview';
    let mouseX = 0, mouseY = 0;
    let targetX = 0, targetY = 0;
    let containerEl = null;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
        } catch (e) {
            return false;
        }
    }

    function initThreeScene(containerId) {
        containerEl = document.getElementById(containerId);
        if (!containerEl) return;

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

            console.info('[AIOD-3D] Three.js Neural Visualizer initialized successfully.');

        } catch (err) {
            console.warn('[AIOD-3D] Three.js initialization caught error:', err);
            containerEl.classList.add('three-fallback');
            dispose();
        }
    }

    function buildNeuralNetwork() {
        neuralSphereGroup = new THREE.Group();
        scene.add(neuralSphereGroup);

        // Adjust node density based on screen width/device capability
        const isMobile = window.innerWidth < 768;
        const nodeCount = isMobile ? 60 : 120;
        const radius = 70;

        const nodePositions = [];
        const colors = [];
        const colorPrimary = new THREE.Color(0x3B82F6); // Electric Blue
        const colorSecondary = new THREE.Color(0x8B5CF6); // Violet
        const colorCyan = new THREE.Color(0x06B6D4); // Cyan

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

            nodePositions.push(px, py, pz);

            // Interpolate colors
            const mixColor = i % 3 === 0 ? colorCyan : (i % 2 === 0 ? colorPrimary : colorSecondary);
            colors.push(mixColor.r, mixColor.g, mixColor.b);
        }

        // Particle Nodes Geometry
        particlesGeometry = new THREE.BufferGeometry();
        particlesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(nodePositions, 3));
        particlesGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const particlesMaterial = new THREE.PointsMaterial({
            size: isMobile ? 3.5 : 4.5,
            vertexColors: true,
            transparent: true,
            opacity: 0.85,
            blending: THREE.AdditiveBlending
        });

        particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
        neuralSphereGroup.add(particlesMesh);

        // Connect nearby nodes with neural synapses (lines)
        const linePositions = [];
        const lineColors = [];
        const maxDist = 32;

        for (let i = 0; i < nodeCount; i++) {
            const x1 = nodePositions[i * 3];
            const y1 = nodePositions[i * 3 + 1];
            const z1 = nodePositions[i * 3 + 2];

            for (let j = i + 1; j < nodeCount; j++) {
                const x2 = nodePositions[j * 3];
                const y2 = nodePositions[j * 3 + 1];
                const z2 = nodePositions[j * 3 + 2];

                const dx = x1 - x2;
                const dy = y1 - y2;
                const dz = z1 - z2;
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

                if (dist < maxDist) {
                    linePositions.push(x1, y1, z1, x2, y2, z2);
                    const alpha = 1.0 - (dist / maxDist);
                    lineColors.push(colorPrimary.r * alpha, colorPrimary.g * alpha, colorPrimary.b * alpha);
                    lineColors.push(colorCyan.r * alpha, colorCyan.g * alpha, colorCyan.b * alpha);
                }
            }
        }

        const linesGeometry = new THREE.BufferGeometry();
        linesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
        linesGeometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));

        const linesMaterial = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.28,
            blending: THREE.AdditiveBlending
        });

        linesMesh = new THREE.LineSegments(linesGeometry, linesMaterial);
        neuralSphereGroup.add(linesMesh);

        // Position sphere slightly offset to the right on desktop for visual balance with hero text
        if (!isMobile) {
            neuralSphereGroup.position.x = 45;
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
            if (containerEl) containerEl.style.opacity = '0.85';
            resume();
        } else {
            // Immediately pause rendering when moving to Detection, Batch, Analytics, or History
            if (containerEl) containerEl.style.opacity = '0.15';
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
        pause: pause,
        resume: resume,
        dispose: dispose
    };

})();

