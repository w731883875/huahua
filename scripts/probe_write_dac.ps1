# Diagnostic: why does a DACL round-trip fail on the workspace folder?
# Prints the exact exception instead of a bare boolean.
param([string]$Path = 'F:\tools\huahua')

$ErrorActionPreference = 'Stop'
$p = (Resolve-Path -LiteralPath $Path).Path
$me = [System.Security.Principal.WindowsIdentity]::GetCurrent()
Write-Host ("folder: {0}" -f $p)
Write-Host ("user  : {0}" -f $me.Name)

try {
    $acl = Get-Acl -LiteralPath $p
    Write-Host ("read ACL: OK  owner={0}  sddl-ish rules={1}" -f $acl.Owner, $acl.Access.Count)
} catch {
    Write-Host ("read ACL FAILED: {0}" -f $_.Exception.GetType().FullName)
    Write-Host $_.Exception.Message
    exit 3
}

try {
    $probe = New-Object System.Security.AccessControl.FileSystemAccessRule(
        $me.User,
        [System.Security.AccessControl.FileSystemRights]'ReadData',
        [System.Security.AccessControl.AccessControlType]::Allow
    )
    $acl.AddAccessRule($probe)
    Set-Acl -LiteralPath $p -AclObject $acl
    Write-Host "write DACL: OK"
} catch {
    Write-Host ("write DACL FAILED: {0}" -f $_.Exception.GetType().FullName)
    Write-Host $_.Exception.Message
    if ($_.Exception.InnerException) {
        Write-Host ("inner: {0}" -f $_.Exception.InnerException.Message)
    }
    exit 4
}

try {
    $restored = Get-Acl -LiteralPath $p
    $restored.RemoveAccessRule($probe) | Out-Null
    Set-Acl -LiteralPath $p -AclObject $restored
    Write-Host "revert DACL: OK"
} catch {
    Write-Host ("revert FAILED: {0}" -f $_.Exception.Message)
    exit 5
}

# Also test the exact call the sandbox makes: SetNamedSecurityInfoW with only
# DACL_SECURITY_INFORMATION, writing the current DACL back unchanged.
$sig = @'
using System;
using System.Runtime.InteropServices;
public static class DaclProbe {
    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    public static extern uint GetNamedSecurityInfoW(
        string pObjectName, int ObjectType, uint SecurityInfo,
        IntPtr ppsidOwner, IntPtr ppsidGroup, out IntPtr ppDacl, IntPtr ppSacl, out IntPtr ppSecurityDescriptor);
    [DllImport("advapi32.dll", SetLastError = true)]
    public static extern uint SetNamedSecurityInfoW(
        string pObjectName, int ObjectType, uint SecurityInfo,
        IntPtr psidOwner, IntPtr psidGroup, IntPtr pDacl, IntPtr pSacl);
    public static uint RewriteDacl(string path) {
        IntPtr dacl, sd;
        uint rc = GetNamedSecurityInfoW(path, 1, 0x00000004, IntPtr.Zero, IntPtr.Zero, out dacl, IntPtr.Zero, out sd);
        if (rc != 0) return rc;
        uint rc2 = SetNamedSecurityInfoW(path, 1, 0x00000004, IntPtr.Zero, IntPtr.Zero, dacl, IntPtr.Zero);
        return rc2;
    }
}
'@
try {
    Add-Type -TypeDefinition $sig -ErrorAction Stop
    $rc = [DaclProbe]::RewriteDacl($p)
    if ($rc -eq 0) {
        Write-Host "SetNamedSecurityInfoW(DACL only): OK (rc=0)  <- the sandbox's own call now succeeds"
    } else {
        Write-Host ("SetNamedSecurityInfoW(DACL only): FAILED rc={0}" -f $rc)
    }
} catch {
    Write-Host ("P/Invoke probe failed: {0}" -f $_.Exception.Message)
}
