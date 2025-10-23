================================================================================
PUSH TO GITHUB - STEP BY STEP
================================================================================

Your local git repository is ready! Here's how to push it to GitHub:

================================================================================
OPTION 1: Create New Repository on GitHub (RECOMMENDED)
================================================================================

**Step 1: Create the Repository on GitHub**
1. Go to: https://github.com/new
2. Repository name: cs378-assignment8-logistic-regression (or your choice)
3. Description: "Spark Logistic Regression for Text Classification (CS378 A8)"
4. Choose: Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

**Step 2: Connect and Push**
After creating the repo, GitHub will show you commands. Use these:

```bash
cd /u/arabel/cloud-computing/A8

# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Rename branch to main (optional, if you prefer main over master)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Replace:**
- YOUR_USERNAME with your GitHub username
- YOUR_REPO_NAME with the repository name you chose

**Example:**
```bash
git remote add origin https://github.com/ArabelRachman/cs378-assignment8.git
git branch -M main
git push -u origin main
```

================================================================================
OPTION 2: Use Existing Repository (If you already have one)
================================================================================

If you want to push to an existing repo:

```bash
cd /u/arabel/cloud-computing/A8

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/EXISTING_REPO.git

# Push to specific branch
git branch -M main
git push -u origin main
```

================================================================================
AUTHENTICATION
================================================================================

When you run `git push`, GitHub will ask for authentication:

**Option A: Personal Access Token (Recommended)**
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" (classic)
3. Give it a name (e.g., "CS378 A8 Upload")
4. Select scopes: at minimum check "repo"
5. Click "Generate token"
6. Copy the token (save it somewhere safe!)
7. When git asks for password, paste the token

**Option B: SSH Key (More secure, one-time setup)**
1. Generate SSH key:
   ```bash
   ssh-keygen -t ed25519 -C "rachman.arabel@gmail.com"
   ```
2. Copy public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
3. Add to GitHub: https://github.com/settings/ssh/new
4. Use SSH URL instead:
   ```bash
   git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git
   ```

================================================================================
QUICK COMMANDS SUMMARY
================================================================================

# 1. Create repo on GitHub first (via web browser)

# 2. Then run these commands:
cd /u/arabel/cloud-computing/A8
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main

# 3. Enter your GitHub username and personal access token when prompted

================================================================================
VERIFY IT WORKED
================================================================================

After pushing, visit:
https://github.com/YOUR_USERNAME/YOUR_REPO_NAME

You should see all your files there!

================================================================================
FUTURE UPDATES
================================================================================

After making changes to files:

```bash
cd /u/arabel/cloud-computing/A8

# Stage changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

================================================================================
CURRENT STATUS
================================================================================

✅ Git repository initialized
✅ Files committed locally
✅ .gitignore configured (excludes logs, data files, temp files)
✅ Ready to push to GitHub

**Next step:** Create repository on GitHub and run the push commands above!

================================================================================
TROUBLESHOOTING
================================================================================

**Error: "remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

**Error: "Permission denied"**
- Make sure you're using the correct authentication method
- Try personal access token instead of password

**Error: "Updates were rejected"**
```bash
git pull origin main --rebase
git push origin main
```

================================================================================
