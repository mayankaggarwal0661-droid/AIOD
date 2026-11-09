/**
 * =====================================================================
 * AIOD ENTERPRISE 3D VISION INTELLIGENCE SYSTEM (Three.js)
 * Concept: "AI Perception Core — A Living Computer Vision Intelligence System"
 *
 * Features:
 * - Central translucent intelligence core (faceted polyhedral lattice & pulsing node)
 * - Interconnected counter-rotating orbital gyro & sensor rings with azimuth graduations
 * - Floating 3D computer vision bounding frames with corner brackets & tracking reticles
 * - Continuous 360-degree laser/radar scanning sweep across the perception field
 * - Orbital data stream particles moving along harmonic curved trajectories
 * - Dynamic theme adaptation (Obsidian Cyber & Noir Gold)
 * - Right-shifted hero composition ensuring unobstructed headline readability
 * - Viewport visibility throttling, reduced-motion compliance & leak-proof disposal
 * =====================================================================
 */

(function () {
    'use strict';

    let scene, camera, renderer, animationFrameId;
    let perceptionCoreGroup;
    let isInitialized = false;
    let isPaused = false;
    let currentSection = 'overview';
    let currentTheme = 'obsidian';
    let mouseX = 0, mouseY = 0;
    let targetX = 0, targetY = 0;
    let baseRotationX = 0, baseRotationY = 0;
    let containerEl = null;

    // Core 3D Elements
    let coreInnerMesh, coreOuterMesh, corePulseNode;
    let ring1Mesh, ring2Mesh, ring3Mesh, reticleMesh, scanSweepMesh;
    let detectionFrames = [];
    let particlesGeometry, particlesMesh, linesGeometry, linesMesh;
    let particleData = [];
    let particleCount = 80;

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
                primary: new THREE.Color(0xD4AF37),        // Classic Metallic Gold
                secondary: new THREE.Color(0xF5D061),      // Radiant Bright Gold
                tertiary: new THREE.Color(0xFFFFFF),       // Pure White Highlight
                coreWireframe: new THREE.Color(0xD4AF37),
                corePulse: new THREE.Color(0xFFFFFF),
                rings: new THREE.Color(0xD4AF37),
                frames: new THREE.Color(0xF5D061),
                framesSecondary: new THREE.Color(0xFFFFFF),
                scanSweep: new THREE.Color(0xD4AF37),
                lineAlphaScale: 0.50
            };
        }
        // Default: Obsidian Cyber
        return {
            primary: new THREE.Color(0x3B82F6),            // Electric Blue
            secondary: new THREE.Color(0x8B5CF6),          // Violet
            tertiary: new THREE.Color(0x06B6D4),           // Cyan
            coreWireframe: new THREE.Color(0x3B82F6),
            corePulse: new THREE.Color(0x06B6D4),
            rings: new THREE.Color(0x3B82F6),
            frames: new THREE.Color(0x06B6D4),
            framesSecondary: new THREE.Color(0x3B82F6),
            scanSweep: new THREE.Color(0x06B6D4),
            lineAlphaScale: 0.38
        };
    }

    function initThreeScene(containerId) {
        containerEl = document.getElementById(containerId);
        if (!containerEl) return;

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

            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
            camera.position.z = 240;

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

            // Build the AI Perception Core
            buildPerceptionCore();

            // Event Listeners
            window.addEventListener('resize', onWindowResize, { passive: true });
            if (!('ontouchstart' in window)) {
                document.addEventListener('pointermove', onPointerMove, { passive: true });
            }
            document.addEventListener('visibilitychange', onVisibilityChange, { passive: true });

            isInitialized = true;

            if (prefersReducedMotion) {
                renderer.render(scene, camera);
                isPaused = true;
            } else {
                startAnimationLoop();
            }

            console.info('[AIOD-3D] AI Perception Core initialized successfully with theme:', currentTheme);

        } catch (err) {
            console.warn('[AIOD-3D] Initialization caught error:', err);
            containerEl.classList.add('three-fallback');
            dispose();
        }
    }

    function buildPerceptionCore() {
        perceptionCoreGroup = new THREE.Group();
        scene.add(perceptionCoreGroup);

        const width = window.innerWidth;
        const isMobile = width < 768;
        const isTablet = width >= 768 && width < 1024;
        const themeColors = getThemeColors(currentTheme);

        // Position primarily towards the right side on desktop so hero text on the left stays legible
        if (isMobile) {
            perceptionCoreGroup.position.set(0, 10, -20);
            perceptionCoreGroup.scale.set(0.62, 0.62, 0.62);
        } else if (isTablet) {
            perceptionCoreGroup.position.set(30, 0, 0);
            perceptionCoreGroup.scale.set(0.85, 0.85, 0.85);
        } else {
            perceptionCoreGroup.position.set(48, 0, 0);
            perceptionCoreGroup.scale.set(1.0, 1.0, 1.0);
        }

        // =================================================================
        // 1. CENTRAL INTELLIGENCE GEOMETRIC CORE
        // =================================================================
        // Inner faceted polyhedron lattice
        const coreInnerGeom = new THREE.IcosahedronGeometry(26, 1);
        const coreInnerMat = new THREE.MeshBasicMaterial({
            color: themeColors.coreWireframe,
            wireframe: true,
            transparent: true,
            opacity: 0.35
        });
        coreInnerMesh = new THREE.Mesh(coreInnerGeom, coreInnerMat);
        perceptionCoreGroup.add(coreInnerMesh);

        // Outer faceted cage shroud
        const coreOuterGeom = new THREE.IcosahedronGeometry(36, 0);
        const coreOuterMat = new THREE.MeshBasicMaterial({
            color: themeColors.secondary,
            wireframe: true,
            transparent: true,
            opacity: 0.20
        });
        coreOuterMesh = new THREE.Mesh(coreOuterGeom, coreOuterMat);
        perceptionCoreGroup.add(coreOuterMesh);

        // Central focal pulse node
        const pulseGeom = new THREE.SphereGeometry(6, 16, 16);
        const pulseMat = new THREE.MeshBasicMaterial({
            color: themeColors.corePulse,
            transparent: true,
            opacity: 0.85
        });
        corePulseNode = new THREE.Mesh(pulseGeom, pulseMat);
        perceptionCoreGroup.add(corePulseNode);

        // =================================================================
        // 2. INTERCONNECTED ORBITAL GYRO & SENSOR RINGS
        // =================================================================
        // Ring 1 (Equatorial Gyro Ring)
        const ring1Geom = new THREE.TorusGeometry(52, 0.35, 6, 80);
        const ring1Mat = new THREE.MeshBasicMaterial({
            color: themeColors.rings,
            transparent: true,
            opacity: 0.38
        });
        ring1Mesh = new THREE.Mesh(ring1Geom, ring1Mat);
        ring1Mesh.rotation.x = Math.PI * 0.25;
        perceptionCoreGroup.add(ring1Mesh);

        // Ring 2 (Polar Sensor Ring)
        const ring2Geom = new THREE.TorusGeometry(70, 0.35, 6, 80);
        const ring2Mat = new THREE.MeshBasicMaterial({
            color: themeColors.secondary,
            transparent: true,
            opacity: 0.30
        });
        ring2Mesh = new THREE.Mesh(ring2Geom, ring2Mat);
        ring2Mesh.rotation.y = Math.PI * 0.35;
        ring2Mesh.rotation.z = Math.PI * 0.15;
        perceptionCoreGroup.add(ring2Mesh);

        // Ring 3 (Outer Azimuth Sensor Ring)
        const ring3Geom = new THREE.TorusGeometry(88, 0.30, 6, 90);
        const ring3Mat = new THREE.MeshBasicMaterial({
            color: themeColors.rings,
            transparent: true,
            opacity: 0.24
        });
        ring3Mesh = new THREE.Mesh(ring3Geom, ring3Mat);
        ring3Mesh.rotation.x = -Math.PI * 0.2;
        perceptionCoreGroup.add(ring3Mesh);

        // Subtle Target Reticle & Coordinate Crosshairs
        const reticleLines = [];
        const rRadius = 46;
        for (let a = 0; a < 64; a++) {
            const theta1 = (a / 64) * Math.PI * 2;
            const theta2 = ((a + 1) / 64) * Math.PI * 2;
            reticleLines.push(Math.cos(theta1) * rRadius, Math.sin(theta1) * rRadius, 0);
            reticleLines.push(Math.cos(theta2) * rRadius, Math.sin(theta2) * rRadius, 0);
        }
        // Crosshair ticks
        const tick = 6;
        reticleLines.push(rRadius - tick, 0, 0, rRadius + tick, 0, 0);
        reticleLines.push(-rRadius - tick, 0, 0, -rRadius + tick, 0, 0);
        reticleLines.push(0, rRadius - tick, 0, 0, rRadius + tick, 0);
        reticleLines.push(0, -rRadius - tick, 0, 0, -rRadius + tick, 0);

        const reticleGeom = new THREE.BufferGeometry();
        reticleGeom.setAttribute('position', new THREE.Float32BufferAttribute(reticleLines, 3));
        const reticleMat = new THREE.LineBasicMaterial({
            color: themeColors.tertiary,
            transparent: true,
            opacity: 0.26
        });
        reticleMesh = new THREE.LineSegments(reticleGeom, reticleMat);
        reticleMesh.rotation.x = Math.PI * 0.28;
        perceptionCoreGroup.add(reticleMesh);

        // =================================================================
        // 3. RADAR / LASER SCANNING SWEEP
        // =================================================================
        const sweepGeom = new THREE.RingGeometry(8, 96, 32, 1, 0, Math.PI * 0.35);
        const sweepMat = new THREE.MeshBasicMaterial({
            color: themeColors.scanSweep,
            transparent: true,
            opacity: 0.12,
            side: THREE.DoubleSide,
            blending: THREE.AdditiveBlending
        });
        scanSweepMesh = new THREE.Mesh(sweepGeom, sweepMat);
        scanSweepMesh.rotation.x = Math.PI * 0.25;
        perceptionCoreGroup.add(scanSweepMesh);

        // =================================================================
        // 4. FLOATING 3D COMPUTER VISION DETECTION FRAMES (Target Brackets)
        // =================================================================
        detectionFrames = [];
        const frameConfigs = isMobile
            ? [
                { pos: [38, 22, -10], size: [20, 16, 12], speed: 1.1, offset: 0 },
                { pos: [-34, -20, 12], size: [18, 18, 14], speed: 0.9, offset: 2 }
            ]
            : [
                { pos: [56, 32, -18], size: [24, 18, 14], speed: 1.2, offset: 0 },
                { pos: [-50, -28, 22], size: [20, 20, 16], speed: 0.9, offset: 1.8 },
                { pos: [44, -36, 16], size: [18, 14, 12], speed: 1.4, offset: 3.2 },
                { pos: [-38, 38, -24], size: [26, 18, 15], speed: 1.0, offset: 4.6 },
                { pos: [18, -52, -10], size: [16, 16, 12], speed: 1.3, offset: 5.5 }
            ];

        frameConfigs.forEach(cfg => {
            const frameObj = createDetectionBoundingBox(cfg.size[0], cfg.size[1], cfg.size[2], themeColors.frames, themeColors.framesSecondary, isMobile);
            frameObj.group.position.set(cfg.pos[0], cfg.pos[1], cfg.pos[2]);
            frameObj.baseX = cfg.pos[0];
            frameObj.baseY = cfg.pos[1];
            frameObj.baseZ = cfg.pos[2];
            frameObj.speed = cfg.speed;
            frameObj.offset = cfg.offset;
            perceptionCoreGroup.add(frameObj.group);
            detectionFrames.push(frameObj);
        });

        // =================================================================
        // 5. ORBITAL DATA STREAM PARTICLES & NEURAL SYNAPSES
        // =================================================================
        particleCount = isMobile ? 36 : (isTablet ? 55 : 80);
        particleData = [];
        const initialPositions = [];
        const initialColors = [];

        for (let i = 0; i < particleCount; i++) {
            const radius = 38 + Math.random() * 56;
            const speed = (0.2 + Math.random() * 0.4) * (Math.random() > 0.5 ? 1 : -1);
            const inclination = (Math.random() - 0.5) * Math.PI * 0.7;
            const phase = Math.random() * Math.PI * 2;

            particleData.push({ radius, speed, inclination, phase, x: 0, y: 0, z: 0 });

            initialPositions.push(0, 0, 0);

            const mixColor = i % 3 === 0 ? themeColors.tertiary : (i % 2 === 0 ? themeColors.primary : themeColors.secondary);
            initialColors.push(mixColor.r, mixColor.g, mixColor.b);
        }

        particlesGeometry = new THREE.BufferGeometry();
        particlesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(initialPositions, 3));
        particlesGeometry.setAttribute('color', new THREE.Float32BufferAttribute(initialColors, 3));

        const particlesMaterial = new THREE.PointsMaterial({
            size: isMobile ? 3.5 : 4.5,
            vertexColors: true,
            transparent: true,
            opacity: 0.88,
            blending: THREE.AdditiveBlending
        });

        particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
        perceptionCoreGroup.add(particlesMesh);

        // Synapse lines connecting nearby particles
        linesGeometry = new THREE.BufferGeometry();
        linesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(particleCount * 6 * 3), 3));
        linesGeometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(particleCount * 6 * 3), 3));

        const linesMaterial = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.35,
            blending: THREE.AdditiveBlending
        });

        linesMesh = new THREE.LineSegments(linesGeometry, linesMaterial);
        perceptionCoreGroup.add(linesMesh);
    }

    /**
     * Helper to construct a volumetric bounding box with iconic computer vision corner brackets [ ]
     */
    function createDetectionBoundingBox(w, h, d, primaryColor, secondaryColor, isMobile) {
        const group = new THREE.Group();
        const lines = [];
        const hw = w / 2, hh = h / 2, hd = d / 2;
        const bl = Math.min(w, h, d) * 0.32; // Corner bracket leg length

        // 8 box corners: [sx, sy, sz]
        const corners = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ];

        // Draw 3 bracket legs at each of the 8 corners
        corners.forEach(([sx, sy, sz]) => {
            const cx = sx * hw, cy = sy * hh, cz = sz * hd;
            // X-leg
            lines.push(cx, cy, cz, cx - sx * bl, cy, cz);
            // Y-leg
            lines.push(cx, cy, cz, cx, cy - sy * bl, cz);
            // Z-leg
            lines.push(cx, cy, cz, cx, cy, cz - sz * bl);
        });

        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
        const mat = new THREE.LineBasicMaterial({
            color: primaryColor,
            transparent: true,
            opacity: 0.55
        });
        const cornerLines = new THREE.LineSegments(geom, mat);
        group.add(cornerLines);

        // Subtle center tracking point
        const ptGeom = new THREE.BufferGeometry();
        ptGeom.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
        const ptMat = new THREE.PointsMaterial({
            color: secondaryColor,
            size: isMobile ? 3 : 4,
            transparent: true,
            opacity: 0.75
        });
        const pt = new THREE.Points(ptGeom, ptMat);
        group.add(pt);

        return { group, cornerLines, ptMat };
    }

    function setTheme(newTheme) {
        currentTheme = newTheme;
        if (!isInitialized) return;

        const themeColors = getThemeColors(newTheme);

        if (coreInnerMesh && coreInnerMesh.material) {
            coreInnerMesh.material.color.set(themeColors.coreWireframe);
        }
        if (coreOuterMesh && coreOuterMesh.material) {
            coreOuterMesh.material.color.set(themeColors.secondary);
        }
        if (corePulseNode && corePulseNode.material) {
            corePulseNode.material.color.set(themeColors.corePulse);
        }

        if (ring1Mesh && ring1Mesh.material) ring1Mesh.material.color.set(themeColors.rings);
        if (ring2Mesh && ring2Mesh.material) ring2Mesh.material.color.set(themeColors.secondary);
        if (ring3Mesh && ring3Mesh.material) ring3Mesh.material.color.set(themeColors.rings);
        if (reticleMesh && reticleMesh.material) reticleMesh.material.color.set(themeColors.tertiary);
        if (scanSweepMesh && scanSweepMesh.material) scanSweepMesh.material.color.set(themeColors.scanSweep);

        // Update detection frames
        detectionFrames.forEach(f => {
            if (f.cornerLines && f.cornerLines.material) {
                f.cornerLines.material.color.set(themeColors.frames);
            }
            if (f.ptMat) {
                f.ptMat.color.set(themeColors.framesSecondary);
            }
        });

        // Update particle colors
        if (particlesGeometry && particlesGeometry.attributes.color) {
            const colorAttr = particlesGeometry.attributes.color;
            for (let i = 0; i < particleCount; i++) {
                const mixColor = i % 3 === 0 ? themeColors.tertiary : (i % 2 === 0 ? themeColors.primary : themeColors.secondary);
                colorAttr.setXYZ(i, mixColor.r, mixColor.g, mixColor.b);
            }
            colorAttr.needsUpdate = true;
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
        if (!camera || !renderer || !perceptionCoreGroup) return;
        const width = window.innerWidth;
        const height = window.innerHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);

        const isMobile = width < 768;
        const isTablet = width >= 768 && width < 1024;
        if (isMobile) {
            perceptionCoreGroup.position.set(0, 10, -20);
            perceptionCoreGroup.scale.set(0.62, 0.62, 0.62);
        } else if (isTablet) {
            perceptionCoreGroup.position.set(30, 0, 0);
            perceptionCoreGroup.scale.set(0.85, 0.85, 0.85);
        } else {
            perceptionCoreGroup.position.set(48, 0, 0);
            perceptionCoreGroup.scale.set(1.0, 1.0, 1.0);
        }
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

        let clock = 0;

        function animate() {
            if (isPaused) {
                animationFrameId = null;
                return;
            }

            animationFrameId = requestAnimationFrame(animate);
            clock += 0.016; // approximate time delta

            // Smooth parallax dampening
            targetX += (mouseX - targetX) * 0.04;
            targetY += (mouseY - targetY) * 0.04;

            baseRotationY += 0.0016;
            baseRotationX += 0.0008;

            if (perceptionCoreGroup) {
                perceptionCoreGroup.rotation.y = baseRotationY + targetX * 0.35;
                perceptionCoreGroup.rotation.x = baseRotationX + targetY * 0.35;
            }

            // Central core rotations & gentle pulse
            if (coreInnerMesh) {
                coreInnerMesh.rotation.y -= 0.003;
                coreInnerMesh.rotation.z += 0.002;
            }
            if (coreOuterMesh) {
                coreOuterMesh.rotation.x += 0.0015;
                coreOuterMesh.rotation.y += 0.0015;
            }
            if (corePulseNode) {
                const pulseScale = 1.0 + Math.sin(clock * 2.5) * 0.18;
                corePulseNode.scale.set(pulseScale, pulseScale, pulseScale);
            }

            // Orbital counter-rotations
            if (ring1Mesh) ring1Mesh.rotation.z += 0.0028;
            if (ring2Mesh) ring2Mesh.rotation.x -= 0.0020;
            if (ring3Mesh) ring3Mesh.rotation.y += 0.0014;
            if (reticleMesh) reticleMesh.rotation.z -= 0.0012;

            // Radar scanning sweep
            if (scanSweepMesh) {
                scanSweepMesh.rotation.z += 0.008;
            }

            // Floating 3D detection frames with harmonic float
            detectionFrames.forEach(f => {
                f.group.position.y = f.baseY + Math.sin(clock * f.speed + f.offset) * 2.8;
                f.group.position.z = f.baseZ + Math.cos(clock * (f.speed * 0.8) + f.offset) * 2.2;
                f.group.rotation.y += 0.002;
            });

            // Update orbital data stream particles
            const themeColors = getThemeColors(currentTheme);
            if (particlesGeometry && particlesGeometry.attributes.position) {
                const posAttr = particlesGeometry.attributes.position;

                for (let i = 0; i < particleCount; i++) {
                    const p = particleData[i];
                    const angle = p.phase + clock * p.speed;
                    p.x = Math.cos(angle) * p.radius;
                    p.y = Math.sin(angle) * p.radius * Math.cos(p.inclination);
                    p.z = Math.sin(angle) * p.radius * Math.sin(p.inclination);
                    posAttr.setXYZ(i, p.x, p.y, p.z);
                }
                posAttr.needsUpdate = true;

                // Update synapse lines between near particles
                if (linesGeometry && linesGeometry.attributes.position) {
                    const linePosAttr = linesGeometry.attributes.position;
                    const lineColorAttr = linesGeometry.attributes.color;
                    let lineIndex = 0;
                    const maxDist = 32;

                    for (let i = 0; i < particleCount; i++) {
                        for (let j = i + 1; j < particleCount; j++) {
                            const dx = particleData[i].x - particleData[j].x;
                            const dy = particleData[i].y - particleData[j].y;
                            const dz = particleData[i].z - particleData[j].z;
                            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

                            if (dist < maxDist && lineIndex < particleCount * 6) {
                                linePosAttr.setXYZ(lineIndex, particleData[i].x, particleData[i].y, particleData[i].z);
                                linePosAttr.setXYZ(lineIndex + 1, particleData[j].x, particleData[j].y, particleData[j].z);

                                const alpha = (1.0 - dist / maxDist) * themeColors.lineAlphaScale;
                                lineColorAttr.setXYZ(lineIndex, themeColors.primary.r * alpha, themeColors.primary.g * alpha, themeColors.primary.b * alpha);
                                lineColorAttr.setXYZ(lineIndex + 1, themeColors.secondary.r * alpha, themeColors.secondary.g * alpha, themeColors.secondary.b * alpha);

                                lineIndex += 2;
                            }
                        }
                    }

                    // Zero-out remaining line segments
                    const totalVertices = linePosAttr.count;
                    for (let k = lineIndex; k < totalVertices; k++) {
                        linePosAttr.setXYZ(k, 0, 0, 0);
                        lineColorAttr.setXYZ(k, 0, 0, 0);
                    }

                    linePosAttr.needsUpdate = true;
                    lineColorAttr.needsUpdate = true;
                }
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
            if (containerEl) containerEl.style.opacity = currentTheme === 'noir-gold' ? '0.78' : '0.85';
            resume();
        } else {
            if (containerEl) containerEl.style.opacity = '0.10';
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
