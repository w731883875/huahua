#Requires -Version 5.1
<#
.SYNOPSIS
    Give the signed-in user full control of one workspace folder, so the DSH
    sandbox can add (and re-add) its own write grant there.

.DESCRIPTION
    Symptom this fixes:

        Error: SetNamedSecurityInfoW failed (Win32 5): grantWrite(F:\tools\huahua)

    The sandbox materialises one deterministic ACE in the workspace DACL before
    it spawns a confined command. Writing a DACL needs WRITE_DAC. A folder whose
    owner is a *different* account (typical for a disk moved from another PC)
    grants ordinary users only Modify through BUILTIN\Users, and Modify does not
    include WRITE_DAC or WRITE_OWNER - so every confined command fails at
    provisioning, even a read-only listing.

    The fix is to make the signed-in user the owner and to grant full control
    with inheritance. Ownership is the part that needs elevation, hence the
    self-elevating launch below. Inheritance (OI)(CI) means descendants pick the
    grant up dynamically, so no recursive walk is needed.

.PARAMETER Path
    The workspace folder. Defaults to the parent of this script's folder.

.PARAMETER Check
    Only report the current owner and whether the user already has WRITE_DAC.
    Changes nothing, needs no elevation.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\fix_workspace_permissions.ps1 -Check

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\fix_workspace_permissions.ps1
#>
[CmdletBinding()]
param(
    [string]$Path,
    [switch]$Check
)

$ErrorActionPreference = 'Stop'

if (-not $Path) {
    $Path = Split-Path -Parent $PSScriptRoot
}
$resolved = (Resolve-Path -LiteralPath $Path).Path.TrimEnd('\')
$me = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$userName = $me.Name
$userSid = $me.User.Value

function Get-OwnerName {
    param([string]$Target)
    try {
        $acl = Get-Acl -LiteralPath $Target
        $ownerSid = $acl.Owner
        try {
            $name = (New-Object System.Security.Principal.SecurityIdentifier($ownerSid)).Translate([System.Security.Principal.NTAccount]).Value
        } catch {
            $name = $ownerSid
        }
        return $name
    } catch {
        return "<unreadable: $($_.Exception.Message)>"
    }
}

function Test-CanWriteDacl {
    param([string]$Target)
    # Opening with WRITE_DAC is exactly what the sandbox does before it grants.
    try {
        $acl = Get-Acl -LiteralPath $Target
        $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
            $userSid,
            [System.Security.AccessControl.FileSystemRights]'ReadData',
            [System.Security.AccessControl.AccessControlType]::Allow
        )
        $acl.AddAccessRule($rule)
        $acl.RemoveAccessRule($rule)
        # In-memory only: we never call Set-Acl here, so nothing is modified.
        return $true
    } catch {
        return $false
    }
}

# The authoritative probe is the call the sandbox itself makes: SetNamedSecurityInfoW
# with DACL_SECURITY_INFORMATION only, writing the current DACL back unchanged. It
# reports a plain Win32 code, so an orphaned SID in the DACL cannot confuse it the
# way .NET's Set-Acl can (that one translates SIDs to names and throws
# IdentityNotMappedException on a folder inherited from another machine's account).
if (-not ('WsDaclProbe' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class WsDaclProbe {
    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern uint GetNamedSecurityInfoW(string pObjectName, int ObjectType, uint SecurityInfo,
        IntPtr ppsidOwner, IntPtr ppsidGroup, out IntPtr ppDacl, IntPtr ppSacl, out IntPtr ppSecurityDescriptor);
    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern uint SetNamedSecurityInfoW(string pObjectName, int ObjectType, uint SecurityInfo,
        IntPtr psidOwner, IntPtr psidGroup, IntPtr pDacl, IntPtr pSacl);
    public static uint RewriteDacl(string path) {
        IntPtr dacl, sd;
        const uint DACL_SECURITY_INFORMATION = 0x00000004;
        const int SE_FILE_OBJECT = 1;
        uint rc = GetNamedSecurityInfoW(path, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
            IntPtr.Zero, IntPtr.Zero, out dacl, IntPtr.Zero, out sd);
        if (rc != 0) { return rc; }
        return SetNamedSecurityInfoW(path, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
            IntPtr.Zero, IntPtr.Zero, dacl, IntPtr.Zero);
    }
}
'@ -ErrorAction Stop
}

function Test-CanOpenWriteDac {
    param([string]$Target)
    try {
        # rc 0 = the DACL was rewritten, i.e. WRITE_DAC is held. 5 = access denied.
        return ([WsDaclProbe]::RewriteDacl($Target) -eq 0)
    } catch {
        return $false
    }
}

function Show-State {
    param([string]$Target)
    $owner = Get-OwnerName -Target $Target
    $canDac = Test-CanOpenWriteDac -Target $Target
    Write-Host ("folder   : {0}" -f $Target)
    Write-Host ("owner    : {0}" -f $owner)
    Write-Host ("signed in: {0}  ({1})" -f $userName, $userSid)
    Write-Host ("WRITE_DAC: {0}" -f $canDac)
    return $canDac
}

Write-Host "=== workspace permission check ==="
$ok = Show-State -Target $resolved

if ($Check) {
    if ($ok) {
        Write-Host "RESULT: OK - the sandbox can provision its grant here."
        exit 0
    }
    Write-Host "RESULT: MISSING - run this script again without -Check (it will ask for administrator)."
    exit 2
}

if ($ok) {
    Write-Host "RESULT: already OK, nothing to do."
    exit 0
}

$isAdmin = ([Security.Principal.WindowsPrincipal]$me).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host ""
    Write-Host "Elevation required: ownership has to change, which needs administrator."
    Write-Host "A UAC prompt will appear - approve it, and this window keeps the result."
    $args = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-NoExit',
        '-File', ('"{0}"' -f $PSCommandPath),
        '-Path', ('"{0}"' -f $resolved)
    )
    Start-Process -FilePath 'powershell.exe' -ArgumentList $args -Verb RunAs | Out-Null
    exit 0
}

Write-Host ""
Write-Host "--- taking ownership ---"
& icacls $resolved /setowner $userName /T /C | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "icacls /setowner returned $LASTEXITCODE (continuing: ownership may already be ours)."
}

Write-Host "--- granting full control (inherited by everything below) ---"
& icacls $resolved /grant ('{0}:(OI)(CI)F' -f $userName) /C | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: icacls /grant returned $LASTEXITCODE"
    exit 1
}

Write-Host ""
Write-Host "=== after ==="
$okAfter = Show-State -Target $resolved
if (-not $okAfter) {
    Write-Host "RESULT: STILL MISSING - check whether another explicit deny overrides the grant."
    exit 1
}
Write-Host "RESULT: FIXED - the sandbox can now provision its grant here."
exit 0
