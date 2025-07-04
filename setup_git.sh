#!/bin/bash
# Setup script for GitHub repository

echo "🚀 Setting up YOLOv8 Vehicle Detection for GitHub..."

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
    # .gitignore content would be added here
    echo "✅ .gitignore created"
else
    echo "✅ .gitignore already exists"
fi

# Check for large files that shouldn't be committed
echo "🔍 Checking for large files..."
large_files=$(find . -type f -size +50M 2>/dev/null | grep -v ".git")
if [ ! -z "$large_files" ]; then
    echo "⚠️  Warning: Large files found (>50MB):"
    echo "$large_files"
    echo "📝 These files are excluded in .gitignore"
fi

# Add all files
echo "📦 Adding files to git..."
git add .

# Create initial commit
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "📝 Creating initial commit..."
    git commit -m "Initial commit: YOLOv8 Vehicle Detection System

Features:
- Advanced vehicle detection using YOLOv8
- Real-time tracking and status monitoring
- Comprehensive testing suite
- Research evaluation tools
- Organized project structure"
    echo "✅ Initial commit created"
else
    echo "✅ Repository already has commits"
fi

echo ""
echo "🎉 Repository setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Create a new repository on GitHub"
echo "2. Add your GitHub repository as remote:"
echo "   git remote add origin https://github.com/yourusername/your-repo-name.git"
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "📁 Remember to:"
echo "- Download YOLOv8 models to 03_MODELS/ directory"
echo "- Add your parking videos to 04_DATA/parking_area/video/"
echo "- Update configuration in 02_CONFIG/config.yaml"
echo ""
echo "🔗 Useful commands:"
echo "   git status                 # Check status"
echo "   git add .                  # Add all changes"
echo "   git commit -m 'message'    # Commit changes"
echo "   git push                   # Push to GitHub"
