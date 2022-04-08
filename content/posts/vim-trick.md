---
title: "Vim Trick"
date: 2022-03-26T13:57:26+08:00
draft: true
---

[TOC]

## How to make local function?
```vim
" file: function.vim
function s:LocalFunction()
  echo "here is local function()"
endfunction
call s:LocalFunction() " ok, calling local function in same script is legal 
```
```vim
" file: main.vim
" call s:LocalFunction() # error, the function is localized
```

## How to mapping this function?
if call the function `s:funcName()`, here may be problem of same function name , so vim use keyword `<SID>funcName()` to replace the function name to correct name implicitly.

```vim
" file: function.vim
function s:LocalFunction()
  echo "here is local function()"
endfunction

" map aaa :call s:LocalFunction()<cr>   
" error, cannot use key `s:xxx()` 
map aaa :call <SID>LocalFunction()<cr>
" ok, using `<SID>` replace `s:` 

command CommandName  call s:LocalFunction() " ok, command need not to care this problem.
autocmd InsertLeave * call s:LocalFunction() " ok, autocmd need not to care this problem.
```
