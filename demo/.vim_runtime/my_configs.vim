" Here's a little trick that uses terminal's bracketed paste mode to
" automatically set/unset Vim's paste mode when you paste
" See https://stackoverflow.com/a/38258720

let &t_SI .= "\<Esc>[?2004h"
let &t_EI .= "\<Esc>[?2004l"

inoremap <special> <expr> <Esc>[200~ XTermPasteBegin()

function! XTermPasteBegin()
  set pastetoggle=<Esc>[201~
  set paste
  return ""
endfunction
