# GEMINI.md - Normalbaum Information Ecosystem (tldraw Agent)

This workspace contains a highly customized **tldraw AI agent** designed to model and analyze **information ecosystems**. It extends the standard tldraw agent template with advanced features for epistemic analysis, structural graph metrics, and agent self-awareness (the **Normalbaum** framework).

## Project Overview

The core of this project is an AI agent that manipulates a [tldraw](https://tldraw.dev) infinite canvas via a chat interface. It models the canvas as an information ecosystem where shapes represent **epistemic claims** and arrows represent **evidential relationships**.

### The Normalbaum Framework
The project is built around the "Normalbaum" (Normal Tree) concept—the idea that aggressive optimization or pruning of information to enforce conformity produces structures that appear healthy but are actually fragile and lacking in genuine diversity.

Key capabilities include:
- **Ecosystem Simulation**: Tagging claims with epistemic statuses (`established`, `contested`, `heterodox`, `suppressed`) and relationship types (`supports`, `contradicts`, `derives-from`).
- **Spectral Analysis**: Computing graph Laplacian eigenvalues and SVD dominance ratios to identify bottlenecks and structural homogenization.
- **Diversity Tracking**: Measuring spatial entropy and color/type distribution to monitor "normalization" tendencies.
- **Epistemic Humility**: Tracking agent overwrites of user content and providing a meta-layer of self-reflection on the power dynamic between agent and user.

## Architecture

The application is a **monorepo-style Vite + React** frontend integrated with a **Cloudflare Worker** backend.

- **`client/`**: React frontend featuring the tldraw canvas, chat panel UI, and agent lifecycle management.
- **`shared/`**: Core logic shared between client and worker, including action utilities, prompt part utilities, shape format converters, and mathematical graph/matrix logic.
- **`worker/`**: Cloudflare Worker backend using **Durable Objects** for model streaming via the **Vercel AI SDK**.

### Key Technologies
- **Frontend**: React, tldraw SDK, Vite
- **Backend**: Cloudflare Workers, Durable Objects, itty-router
- **AI**: Vercel AI SDK (`ai` package), supporting Anthropic (Claude), Google (Gemini), and OpenAI.
- **Math**: Custom graph theory and matrix math (Laplacian, Eigenvalues, SVD) in `shared/math/`.

## Development Commands

```bash
# Install dependencies
npm install

# Run development server (Local: http://localhost:5173/)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Configuration & Environment

Create a `.dev.vars` file in the root directory with your API keys:
```env
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
OPENAI_API_KEY=your_key
```

## Key Extension Points

Most customizations happen in `shared/AgentUtils.ts`, which registers:

- **`PROMPT_PART_UTILS`**: Determines what the agent **sees** (e.g., `EcosystemPartUtil`, `DiversityPartUtil`, `ScreenshotPartUtil`).
- **`AGENT_ACTION_UTILS`**: Determines what the agent can **do** (e.g., `SetEpistemicStatusActionUtil`, `SpectralAnalysisActionUtil`, `PruneActionUtil`).

### Mathematical Submodels
The agent uses specialized math utilities in `shared/math/`:
- `GraphMatrix.ts`: Adjacency, Laplacian, and weighted matrix construction.
- `Eigen.ts`: Symmetric eigenvalue decomposition and SVD.

## Directory Structure Notes

The root directory contains many peripheral files (STL 3D models, Excel data, Mathematica notebooks) that appear to be part of a broader research context (likely "Evolutionary Robotics") but are not core to the tldraw agent code itself.

The primary codebase resides in:
- `client/`
- `shared/`
- `worker/`

## Documentation

- `README.md`: Basic project intro.
- `CLAUDE.md`: Detailed architecture and request flow guide for AI coding assistants.
- `ODD_tldraw_agent.md`: Comprehensive model description using the ODD (Overview, Design concepts, Details) protocol.
- `NormalBaum.txt`: Detailed breakdown of the Normalbaum features and design philosophy.
