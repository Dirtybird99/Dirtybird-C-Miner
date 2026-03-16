// Auto-generated vertical SIMD ops. val and pos2_vals are __m256i.
switch(op) {
  case 0: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 1: {
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 2: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_reverse_epi8(val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 3: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 4: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 5: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 6: {
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 7: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 8: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 9: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 10: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 11: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 12: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 13: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 14: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 15: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 16: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 17: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 18: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 19: {
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 20: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 21: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_add_epi8(val, val);
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 22: {
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_reverse_epi8(val);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 23: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 24: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 25: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 26: {
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_add_epi8(val, val);
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 27: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 28: {
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_add_epi8(val, val);
    val = _mm256_add_epi8(val, val);
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 29: {
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 30: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 31: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 32: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_reverse_epi8(val);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 33: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_reverse_epi8(val);
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 34: {
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 35: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 36: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 37: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 38: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 39: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 40: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 41: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 42: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 43: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_add_epi8(val, val);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 44: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 45: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 46: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_add_epi8(val, val);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 47: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 48: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 49: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_add_epi8(val, val);
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 50: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_add_epi8(val, val);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 51: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 52: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 53: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 54: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 55: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 56: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 57: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 58: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 59: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 60: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_rol_epi8(val, 3);
    break;
  }
  case 61: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 62: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 63: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 64: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 65: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 66: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 67: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 68: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 69: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_reverse_epi8(val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 70: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 71: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 72: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 73: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_reverse_epi8(val);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 74: {
    val = _mm256_mul_epi8(val, val);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_reverse_epi8(val);
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 75: {
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 76: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 77: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_add_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 78: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_reverse_epi8(val);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 79: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_add_epi8(val, val);
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 80: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_add_epi8(val, val);
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 81: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 82: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 83: {
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_reverse_epi8(val);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 84: {
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 85: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 86: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 87: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 88: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 89: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 90: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 91: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 92: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 93: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 94: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 95: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 96: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 97: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 98: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 99: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_reverse_epi8(val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 100: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 101: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 102: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_add_epi8(val, val);
    val = _mm256_rol_epi8(val, 3);
    break;
  }
  case 103: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 104: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 105: {
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 106: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 107: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 108: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 109: {
    val = _mm256_mul_epi8(val, val);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 110: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 111: {
    val = _mm256_mul_epi8(val, val);
    val = _mm256_reverse_epi8(val);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 112: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 113: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 114: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_reverse_epi8(val);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 115: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 3);
    break;
  }
  case 116: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 117: {
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 118: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_add_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 119: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 120: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 121: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_add_epi8(val, val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 122: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 123: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_rol_epi8(val, 3);
    break;
  }
  case 124: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 125: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_add_epi8(val, val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 126: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 127: {
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 128: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 129: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 130: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 131: {
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 132: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_reverse_epi8(val);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 133: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 134: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 135: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_add_epi8(val, val);
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 136: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 137: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_reverse_epi8(val);
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 138: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_add_epi8(val, val);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 139: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 3);
    break;
  }
  case 140: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 141: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 142: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 143: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 144: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 145: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 146: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 147: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 148: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 149: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_reverse_epi8(val);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 150: {
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 151: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 152: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 153: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 154: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 155: {
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 156: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 157: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 158: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_add_epi8(val, val);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 159: {
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 160: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_reverse_epi8(val);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_rol_epi8(val, 3);
    break;
  }
  case 161: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 162: {
    val = _mm256_mul_epi8(val, val);
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 163: {
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 164: {
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 165: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 166: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_add_epi8(val, val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 167: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 168: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 169: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 170: {
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_reverse_epi8(val);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 171: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 172: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 173: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 174: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 175: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 176: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 177: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 178: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_add_epi8(val, val);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 179: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_add_epi8(val, val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 180: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 181: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 182: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 183: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 184: {
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 185: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 186: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 187: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_add_epi8(val, val);
    val = _mm256_rol_epi8(val, 3);
    break;
  }
  case 188: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 189: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 190: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 191: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 192: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_add_epi8(val, val);
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 193: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 194: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 195: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 196: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_reverse_epi8(val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 197: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 198: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_reverse_epi8(val);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 199: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_add_epi8(val, val);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 200: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_reverse_epi8(val);
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 201: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 202: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 203: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 204: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 205: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 206: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_reverse_epi8(val);
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 207: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 208: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_add_epi8(val, val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 3);
    break;
  }
  case 209: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 210: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 211: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_add_epi8(val, val);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 212: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 213: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 214: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 215: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 216: {
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 217: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_add_epi8(val, val);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 218: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 219: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 220: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_reverse_epi8(val);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 221: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 222: {
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_mul_epi8(val, val);
    break;
  }
  case 223: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 224: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 225: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_reverse_epi8(val);
    val = _mm256_rol_epi8(val, 3);
    break;
  }
  case 226: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 227: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 228: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_add_epi8(val, val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 229: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 230: {
    val = _mm256_mul_epi8(val, val);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 231: {
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_reverse_epi8(val);
    break;
  }
  case 232: {
    val = _mm256_mul_epi8(val, val);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 233: {
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    break;
  }
  case 234: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 235: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_mul_epi8(val, val);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 236: {
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_add_epi8(val, val);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 237: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 3);
    break;
  }
  case 238: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_add_epi8(val, val);
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    break;
  }
  case 239: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_mul_epi8(val, val);
    val = _mm256_and_si256(val, pos2_vals);
    break;
  }
  case 240: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_add_epi8(val, val);
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 241: {
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, pos2_vals);
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 242: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_add_epi8(val, val);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_xor_si256(val, pos2_vals);
    break;
  }
  case 243: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rol_epi8(val, 1);
    break;
  }
  case 244: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_reverse_epi8(val);
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 245: {
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 246: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_rol_epi8(val, 1);
    val = _mm256_srlv_epi8(val, _mm256_and_si256(val, vec_3));
    val = _mm256_add_epi8(val, val);
    break;
  }
  case 247: {
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 5);
    val = _mm256_xor_si256(val, vec_ff);
    break;
  }
  case 248: {
    val = _mm256_xor_si256(val, vec_ff);
    val = _mm256_sub_epi8(val, _mm256_xor_si256(val, vec_97));
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rol_epi8(val, 5);
    break;
  }
  case 249: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_rolv_epi8(val, val);
    break;
  }
  case 250: {
    val = _mm256_and_si256(val, pos2_vals);
    val = _mm256_rolv_epi8(val, val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    break;
  }
  case 251: {
    val = _mm256_add_epi8(val, val);
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    break;
  }
  case 252: {
    val = _mm256_reverse_epi8(val);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 4));
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_sllv_epi8(val, _mm256_and_si256(val, vec_3));
    break;
  }
  case 253: {
    val = _mm256_rol_epi8(val, 3);
    break;
  }
  case 254: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 3);
    break;
  }
  case 255: {
    val = _mm256_xor_si256(val, popcnt256_epi8(val));
    val = _mm256_rol_epi8(val, 3);
    val = _mm256_xor_si256(val, _mm256_rol_epi8(val, 2));
    val = _mm256_rol_epi8(val, 3);
    break;
  }
}
