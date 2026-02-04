# EMODB: Emotion Analysis

## Step 1: Data Prepration

1. Split the raw audio(.wav) files into parts

```bash
// Conditions to split
if duration>=6:
    parts = 4
elif duration>=4:
    parts = 3
elif duration>=2:
    parts = 2
else parts = 1
```
